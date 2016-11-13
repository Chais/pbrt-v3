//
// Created by chais on 05/11/16.
//

#include "rvpl.h"

void RichVPLIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    if (scene.lights.size() == 0) return;
    std::unique_ptr<Sampler> sampler1 = sampler.Clone(168411684);
    MemoryArena arena;
    RNG rng(13);
    vlSetOffset = uint32_t(std::round(rng.UniformFloat() * nLightSets)) % nLightSets;

    // Compute samples for rays emitted from lights
    int nSamples = nLightPaths * nLightSets;
    sampler1->StartPixel(Point2i());
    sampler1->Request1DArray(nSamples);
    const Float *lightNum = sampler1->Get1DArray(nSamples);
    //sampler1->StartNextSample();
    sampler1->Request2DArray(nSamples);
    const Point2f *lightSampPos = sampler1->Get2DArray(nSamples);
    //sampler1->StartNextSample();
    sampler1->Request2DArray(nSamples);
    const Point2f *lightSampDir = sampler1->Get2DArray(nSamples);
    sampler1->Request2DArray(nSamples);
    const Point2f *bsdfSamples = sampler1->Get2DArray(nSamples);
    //int bsdfSample = 0;

    // Compute light sampling densities
    Distribution1D lightDistribution = *ComputeLightPowerDistribution(scene);
    for (uint32_t s = 0; s < nLightSets; s++) {
        for (uint32_t i = 0; i < nLightPaths; i++) {
            // Follow path _i_ from light to create virtual lights
            int sampOffset = s * nLightPaths + i;

            // Choose light source to trace virtual light path from
            Float lightPdf;
            int ln = lightDistribution.SampleDiscrete(sampler1->Get1D(), &lightPdf);
            const std::shared_ptr<Light> &light = scene.lights[ln];

            // Sample ray leaving light source for virtual light path
            RayDifferential ray;
            Float pdfPos, pdfDir;
            Normal3f nLight;
            Spectrum alpha = light->Sample_Le(sampler1->Get2D(), sampler1->Get2D(), 0, &ray, &nLight,
                                              &pdfPos, &pdfDir);
            if (pdfPos == 0.f || pdfDir == 0.f || alpha.IsBlack()) continue;
            alpha *= AbsDot(nLight, ray.d) / (pdfDir * pdfPos * lightPdf);
            SurfaceInteraction isect;
            while (scene.Intersect(ray, &isect) && !alpha.IsBlack()) {
                // Attenuate for participating medium
                if (ray.medium) {
                    MediumInteraction mi;
                    alpha *= ray.medium->Sample(ray, *sampler1, arena, &mi);
                }

                // Create virtual light at ray intersection point
                Vector3f wo = isect.wo;
                isect.ComputeScatteringFunctions(ray, arena);
                BSDF *bsdf = isect.bsdf;
                Point2f bsdfSample = sampler1->Get2D();
                Spectrum contrib = alpha * bsdf->rho(wo, 1, &bsdfSample) * InvPi;
                virtualLights[s].push_back(VirtualLight(isect.p, isect.n, contrib, 5e-4f * isect.time));

                // Sample new ray direction and update weight for virtual light path
                Vector3f wi;
                Float pdf;
                Spectrum fr = isect.bsdf->Sample_f(wo, &wi, sampler1->Get2D(), &pdf);
                if (fr.IsBlack() || pdf == 0.f) break;
                Spectrum contribScale = fr * AbsDot(wi, isect.n) / pdf;

                // Possibly terminate virtual light path with Russian roulette
                Float rrProb = std::min(1.f, contribScale.y());
                if (rng.UniformFloat() > rrProb) break;
                alpha *= contribScale / rrProb;
                ray = RayDifferential(isect.p, wi);
            }
            arena.Reset();
        }
    }
    if (strategy == LightStrategy::UniformSampleAll) {
        // Compute number of samples to use for each light
        for (const auto &light : scene.lights)
            nLightSamples.push_back(sampler.RoundCount(light->nSamples));

        // Request samples for sampling all lights
        for (int i = 0; i < maxDepth; ++i) {
            for (size_t j = 0; j < scene.lights.size(); ++j) {
                sampler.Request2DArray(nLightSamples[j]);
            }
        }
    }
    return;
}

Spectrum RichVPLIntegrator::Li(const RayDifferential &ray, const Scene &scene, Sampler &sampler, MemoryArena &arena,
                               int depth) const {
    ProfilePhase pp(Prof::SamplerIntegratorLi);
    Spectrum L(0.f);
    if (scene.lights.size() == 0) return L;

    // Find closest ray intersection or return background radiance
    SurfaceInteraction isect;
    if (!scene.Intersect(ray, &isect)) {
        for (const auto &light : scene.lights) L += light->Le(ray);
        return L;
    }

    // Compute scattering functions for surface interaction
    isect.ComputeScatteringFunctions(ray, arena);
    if (!isect.bsdf) return Li(isect.SpawnRay(ray.d), scene, sampler, arena, depth);

    // Compute emitted light if ray hit an area light source
    Vector3f wo = isect.wo;
    L += isect.Le(wo);
    // Compute direct lighting for _DirectLightingIntegrator_ integrator
    if (strategy == LightStrategy::UniformSampleAll)
        L += UniformSampleAllLights(isect, scene, arena, sampler, nLightSamples);
    else
        L += UniformSampleOneLight(isect, scene, arena, sampler);

    const Point3f &p = isect.p;
    const Normal3f &n = isect.n;
    // Compute indirect illumination with virtual lights
    uint32_t lSet = (vlSetOffset + sampler.CurrentSampleNumber()) % nLightSets;
    for (uint32_t i = 0; i < virtualLights[lSet].size(); i++) {
        const VirtualLight &vl = virtualLights[lSet][i];
        // Compute virtual light's tentative contribution _Llight_
        Float d2 = DistanceSquared(p, vl.p);
        Vector3f wi = Normalize(vl.p - p);
        Float G = AbsDot(wi, n) * AbsDot(wi, vl.n) / d2;
        G = std::min(G, gLimit);
        Spectrum f = isect.bsdf->f(wo, wi);
        if (G == 0.f || f.IsBlack()) continue;
        Spectrum Llight = f * G * vl.pathContrib / nLightPaths;
        RayDifferential connectRay(p, wi, std::sqrt(d2) * (1.f - vl.rayEpsilon), vl.rayEpsilon, ray.medium);
        if (connectRay.medium) {
            MediumInteraction mi;
            Llight *= connectRay.medium->Sample(ray, sampler, arena, &mi);
        }

        // Possible skip virtual light shadow ray with Russian roulette
        if (Llight.y() < rrThreshold) {
            Float continueProbability = .1f;
            if (sampler.Get1D() > continueProbability) continue;
            Llight /= continueProbability;
        }

        // Add contribution from _VirtualLight_ _vl_
        if (!scene.IntersectP(connectRay)) L += Llight;
    }
    if (depth < maxDepth) {
        // Do bias compensation for bounding geometry term
        int nSamples = (depth == 0) ? nGatherSamples : 1;
        for (int i = 0; i < nSamples; i++) {
            Vector3f wi;
            Float pdf;
            Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf, BxDFType(BSDF_ALL & ~BSDF_SPECULAR));
            if (!f.IsBlack() && pdf > 0.f) {
                // Trace ray for bias compensation gather sample
                Float maxDist = std::sqrt(AbsDot(wi, n) / gLimit);
                RayDifferential gatherRay(p, wi, maxDist, isect.pError.Length());
                Spectrum Li = this->Li(gatherRay, scene, sampler, arena, depth+1);
                if (Li.IsBlack()) continue;

                // Add bias compensation ray contribution to radiance sum
                SurfaceInteraction gatherIsect;
                scene.Intersect(gatherRay, &gatherIsect);
                Float Ggather = AbsDot(wi, n) * AbsDot(-wi, gatherIsect.n) / DistanceSquared(p, gatherIsect.p);
                if (Ggather - gLimit > 0.f && Ggather != Infinity) {
                    Float gs = (Ggather - gLimit) / Ggather;
                    L += f * Li * (AbsDot(wi ,n) * gs / (nSamples * pdf));
                }
            }
        }
    }
    if (depth + 1 < maxDepth) {
        // Trace rays for specular reflection and refraction
        L += SpecularReflect(ray, isect, scene, sampler, arena, depth);
        L += SpecularTransmit(ray, isect, scene, sampler, arena, depth);
    }
    return L;
}

RichVPLIntegrator *
CreateRVPLIntegrator(const ParamSet &params, std::shared_ptr<Sampler> sampler, std::shared_ptr<const Camera> camera) {
    int nLightPaths = params.FindOneInt("nlights", 64);
    Assert(nLightPaths > 0);
    if (PbrtOptions.quickRender) nLightPaths = std::max(1, nLightPaths / 4);
    int nLightSets = params.FindOneInt("nsets", 4);
    Assert(nLightSets > 0);
    float gLimit = params.FindOneFloat("glimit", 10.f);
    int nGatherSamples = params.FindOneInt("gathersamples", 16);
    float rrThreshold = params.FindOneFloat("rrthreshold", .0001f);
    int maxDepth = params.FindOneInt("maxdepth", 5);
    LightStrategy strategy;
    std::string st = params.FindOneString("strategy", "all");
    if (st == "one")
        strategy = LightStrategy::UniformSampleOne;
    else if (st == "all")
        strategy = LightStrategy::UniformSampleAll;
    else {
        Warning("Strategy \"%s\" for direct lighting unknown. Using \"all\".", st.c_str());
        strategy = LightStrategy::UniformSampleAll;
    }
    return new RichVPLIntegrator(camera, sampler, nLightPaths, nLightSets, gLimit, nGatherSamples, rrThreshold,
                                 maxDepth,
                                 strategy);
}
