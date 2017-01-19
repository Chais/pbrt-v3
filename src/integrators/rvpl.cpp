//
// Created by Philip Abernethy (1206672) on 05/11/16.
//

#include "rvpl.h"

void RichVPLIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    if (scene.lights.size() == 0) return;
    MemoryArena arena;
    sampler.StartPixel(Point2i());
    RNG rng(13);
    vlSetOffset = uint32_t(std::round(rng.UniformFloat() * nLightSets)) % nLightSets;

    // Compute light sampling densities
    Distribution1D lightDistribution = *ComputeLightPowerDistribution(scene);
    for (uint32_t s = 0; s < nLightSets; s++) {
        for (uint32_t i = 0; i < nLightPaths; i++) {
            // Choose light source to trace virtual light path from
            Float lightPdf;
            int ln = lightDistribution.SampleDiscrete(sampler.Get1D(), &lightPdf);
            const std::shared_ptr<Light> &light = scene.lights[ln];

            // Sample ray leaving light source for virtual light path
            RayDifferential ray;
            Float pdfPos, pdfDir;
            Normal3f nLight;
            // Light emitted at light source
            Spectrum alpha = light->Sample_Le(sampler.Get2D(), sampler.Get2D(), ray.time, &ray, &nLight, &pdfPos,
                                              &pdfDir);
            if (pdfPos == 0.f || pdfDir == 0.f || alpha.IsBlack()) continue;
            alpha *= AbsDot(nLight, ray.d) / (pdfDir * pdfPos * lightPdf); // Cosine weighting for area lights
            SurfaceInteraction isect;
            while (scene.Intersect(ray, &isect) && !alpha.IsBlack()) {
                // Attenuate for participating medium
                if (ray.medium) alpha *= ray.medium->Tr(ray, sampler);

                // Create virtual light at ray intersection point
                Vector3f wo = isect.wo;
                isect.ComputeScatteringFunctions(ray, arena);
                std::shared_ptr<VirtualLight> vl(
                        new VirtualLight(OffsetRayOrigin(isect.p, isect.pError, isect.n, isect.wo), isect.n));
                // Orient sphere correctly
                Vector3f axis = Cross(Vector3f(0, 0, 1), Vector3f(isect.n));
                if (axis == Vector3f()) axis = Vector3f(1, 0, 0);
                vl->trans = std::make_shared<Transform>(Translate(Vector3f(vl->p)) * Rotate(Degrees(
                        std::acos(Dot(Vector3f(0, 0, 1), Vector3f(isect.n)))), axis));
                vl->itrans = std::make_shared<Transform>(Inverse(*vl->trans));
                // Sample outgoing light for every texel
                Spectrum data[vlResolution.x * vlResolution.y];
                int32_t yres = vlResolution.y / 2;
                for (uint32_t theta = 0; theta < yres; theta++) {
                    Point2f sDir(theta * vlTexelStep.y + vlTexelOffset.y, vlTexelOffset.x);
                    for (uint32_t phi = 0; phi < vlResolution.x; phi++, sDir.y += vlTexelStep.x) {
                        Vector3f wi = (*vl->trans)(UniformSampleHemisphere(sDir));
                        data[(yres + theta) * vlResolution.x + phi] = alpha * isect.bsdf->f(wo, wi); // $f(p, \omega_o, \omega_i)
                    }
                }
                vl->mipmap.reset(new MIPMap<Spectrum>(vlResolution, data, true));
                std::unique_ptr<TextureMapping2D> map;
                map.reset(new UVMapping2D());
                vl->pathContrib.reset(new ImageTexture<Spectrum, Spectrum>(std::move(map), vl->mipmap.get()));
                vl->sphere.reset(new Sphere(vl->trans.get(), vl->itrans.get(), false, .2f, -1, 1, 360));
                virtualLights[s].push_back(vl);
                // Sample new ray direction and update weight for virtual light path
                Vector3f wi;
                Float pdf;
                Spectrum fr = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf);
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
    for (std::vector<std::shared_ptr<VirtualLight>> v : virtualLights)
        nVirtualLights += v.size();
    if (strategy == LightStrategy::UniformSampleAll) {
        // Compute number of samples to use for each light
        for (const auto &light : scene.lights)
            nLightSamples.push_back(sampler.RoundCount(light->nSamples));

        // Request samples for sampling all lights
        for (int i = 0; i < maxDepth; ++i)
            for (size_t j = 0; j < scene.lights.size(); ++j)
                sampler.Request2DArray(nLightSamples[j]);
    }
    return;
}

Spectrum RichVPLIntegrator::Li(const RayDifferential &ray, const Scene &scene, Sampler &sampler, MemoryArena &arena,
                               int depth) const {
    ProfilePhase pp(Prof::SamplerIntegratorLi);
    Spectrum L(0.f);
    if (scene.lights.size() == 0) return L;
    SurfaceInteraction isect;
    uint32_t lSet = (vlSetOffset + sampler.CurrentSampleNumber()) % nLightSets;

    // Show virtual lights instead if configured
    if (showVLights && depth == 0) {
        Float d = Infinity;
        for (uint32_t s = 0; s < nLightSets; s++)
            for (uint32_t i = 0; i < virtualLights[s].size(); i++) {
                Float tHit = Infinity;
                virtualLights[s][i]->sphere->Intersect(ray, &tHit, &isect, false);
                if (tHit < d) {
                    d = tHit;
                    L = virtualLights[s][i]->pathContrib->Evaluate(isect) / nLightPaths;
                }
            }
        if (d < Infinity) return L;
    }

    // Find closest ray intersection or return background radiance
    if (!scene.Intersect(ray, &isect)) {
        for (const auto &light : scene.lights) L += light->Le(ray);
        return L;
    }

    // Compute scattering functions for surface interaction
    isect.ComputeScatteringFunctions(ray, arena);
    if (!isect.bsdf) return Li(isect.SpawnRay(ray.d), scene, sampler, arena, depth + 1);

    // Compute emitted light if ray hit an area light source
    Vector3f wo = isect.wo;
    L += isect.Le(wo);

    // Calculate direct lighting if desired
    if (!noDirectLighting) {
        // Compute direct lighting for _DirectLightingIntegrator_ integrator
        if (strategy == LightStrategy::UniformSampleAll)
            L += UniformSampleAllLights(isect, scene, arena, sampler, nLightSamples);
        else
            L += UniformSampleOneLight(isect, scene, arena, sampler);
    }

    const Point3f &p = OffsetRayOrigin(isect.p, isect.pError, isect.n, isect.wo);
    const Normal3f &n = isect.n;

    // Compute indirect illumination with virtual lights
    for (uint32_t i = 0; i < virtualLights[lSet].size(); i++) {
        const VirtualLight &vl = *virtualLights[lSet][i];
        // Compute virtual light's tentative contribution _Llight_
        Float d2 = DistanceSquared(p, vl.p);
        Vector3f wi = Normalize(vl.p - p);
        Float G = AbsDot(wi, n) * AbsDot(wi, vl.n) / d2;
        G = std::min(G, gLimit);
        Spectrum f = isect.bsdf->f(wo, wi);
        if (G == 0.f || f.IsBlack()) continue;
        //Sphere sphere = Sphere(&VLTransforms[lSet][i], &VLITransforms[lSet][i], false, .05f, 0, 1, 360);
        SurfaceInteraction lisect;
        Float tHit;
        vl.sphere->Intersect(Ray(vl.p, -wi), &tHit, &lisect, false);
        Spectrum Llight = f * G * vl.pathContrib->Evaluate(lisect) / nLightPaths;
        RayDifferential connectRay(p, wi, std::sqrt(d2), isect.time, ray.medium);
        if (connectRay.medium) Llight *= connectRay.medium->Tr(ray, sampler);

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
                RayDifferential gatherRay(p, wi, maxDist);
                Spectrum Li = this->Li(gatherRay, scene, sampler, arena, depth + 1);
                if (Li.IsBlack()) continue;

                // Add bias compensation ray contribution to radiance sum
                SurfaceInteraction gatherIsect;
                scene.Intersect(gatherRay, &gatherIsect);
                Float Ggather = AbsDot(wi, n) * AbsDot(-wi, gatherIsect.n) / DistanceSquared(p, gatherIsect.p);
                if (Ggather - gLimit > 0.f && Ggather != Infinity) {
                    Float gs = (Ggather - gLimit) / Ggather;
                    L += f * Li * (AbsDot(wi, n) * gs / (nSamples * pdf));
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
    int vres = params.FindOneInt("vres", 4);
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
    bool vl = params.FindOneBool("showvl", false);
    bool dl = params.FindOneBool("nodl", false);
    return new RichVPLIntegrator(camera, sampler, nLightPaths, nLightSets, gLimit, nGatherSamples, rrThreshold,
                                 maxDepth, vres, strategy, vl, dl);
}
