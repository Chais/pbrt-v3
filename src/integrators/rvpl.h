//
// Created by Philip Abernethy (1206672) on 05/11/16.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_RVPL_H
#define PBRT_INTEGRATORS_RVPL_H

#include <lights/point.h>
#include <core/stats.h>
#include <core/sampling.h>
#include <shapes/sphere.h>
#include <core/interaction.h>
#include <textures/imagemap.h>
#include "integrator.h"
#include "paramset.h"
#include "directlighting.h"

STAT_COUNTER("Scene/VirtualLights created", nVirtualLights);

struct VirtualLight {
    VirtualLight() {}

    VirtualLight(const Point3f &pp, const Normal3f &nn) : p(pp), n(nn) {}

    VirtualLight(const VirtualLight &vl) {}

    Point3f p;
    Normal3f n;
    // TODO: Store SurfaceInteraction and incoming light instead of texture?
    std::shared_ptr<ImageTexture<Spectrum, Spectrum>> pathContrib;
    std::shared_ptr<Sphere> sphere;
    std::unique_ptr<MIPMap<Spectrum>> mipmap;
    std::shared_ptr<Transform> trans;
    std::shared_ptr<Transform> itrans;
};

class RichVPLIntegrator : public SamplerIntegrator {
public:
    RichVPLIntegrator(const std::shared_ptr<const Camera> &camera, const std::shared_ptr<Sampler> &sampler,
                      const uint32_t &nl, const uint32_t &ns, const float &gl, const int &ng, const float &rrt,
                      const int &maxd, const uint32_t &vres, const LightStrategy &strat, const bool &vl, const bool &dl)
            : SamplerIntegrator(camera, sampler),
              nLightPaths(RoundUpPow2(int32_t(nl))), nLightSets(RoundUpPow2(int32_t(ns))), gLimit(gl),
              nGatherSamples(ng), rrThreshold(rrt), maxDepth(maxd), strategy(strat), showVLights(vl),
              noDirectLighting(dl), vlResolution(4 * RoundUpPow2(int32_t(vres)), 2 * RoundUpPow2(int32_t(vres))) {
        vlTexelStep = Point2f(2.f / vlResolution.x, 2.f / vlResolution.y);
        vlTexelOffset = Point2f(vlTexelStep.x / 2, vlTexelStep.y / 2);
        virtualLights.resize(ns);
    }

    virtual void Preprocess(const Scene &scene, Sampler &s) override;

    virtual Spectrum
    Li(const RayDifferential &ray, const Scene &scene, Sampler &sampler, MemoryArena &arena, int depth) const override;

private:
    uint32_t nLightPaths, nLightSets;
    const Float gLimit;
    const int nGatherSamples;
    const Float rrThreshold;
    const int maxDepth;
    const LightStrategy strategy;
    const bool showVLights;
    const bool noDirectLighting;
    const Point2i vlResolution;
    Point2f vlTexelOffset;
    Point2f vlTexelStep;
    std::vector<std::vector<std::shared_ptr<VirtualLight>>> virtualLights;
    uint32_t vlSetOffset;
    std::vector<int> nLightSamples;
};

RichVPLIntegrator *
CreateRVPLIntegrator(const ParamSet &params, std::shared_ptr<Sampler> sampler, std::shared_ptr<const Camera> camera);

#endif //PBRT_INTEGRATORS_RVPL_H