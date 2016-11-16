//
// Created by Philip Abernethy (1206672) on 05/11/16.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_VPL_H
#define PBRT_INTEGRATORS_VPL_H

#include <lights/point.h>
#include <core/stats.h>
#include <core/sampling.h>
#include <shapes/sphere.h>
#include "integrator.h"
#include "paramset.h"
#include "directlighting.h"

struct VirtualLight {
    VirtualLight() {}

    VirtualLight(const Point3f &pp, const Normal3f &nn, const Spectrum &c) : p(pp), n(nn), pathContrib(c) {}

    Point3f p;
    Normal3f n;
    Spectrum pathContrib;
};

class RichVPLIntegrator : public SamplerIntegrator {
public:
    RichVPLIntegrator(const std::shared_ptr<const Camera> &camera, const std::shared_ptr<Sampler> &sampler, uint32_t nl,
                      uint32_t ns, const float gl, const int ng, const float rrt, const int maxd,
                      const LightStrategy strat, const bool vl) : SamplerIntegrator(camera, sampler),
                                                                  nLightPaths(RoundUpPow2(int32_t(nl))),
                                                                  nLightSets(RoundUpPow2(int32_t(ns))), gLimit(gl),
                                                                  nGatherSamples(ng), rrThreshold(rrt), maxDepth(maxd),
                                                                  strategy(strat), showVLights(vl) {
        virtualLights.resize(ns);
        if (vl) {
            VLTransforms.resize(ns);
            VLITransforms.resize(ns);
        }
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
    std::vector<std::vector<VirtualLight>> virtualLights;
    std::vector<std::vector<Transform>> VLTransforms;
    std::vector<std::vector<Transform>> VLITransforms;
    uint32_t vlSetOffset;
    std::vector<int> nLightSamples;
};

RichVPLIntegrator *
CreateRVPLIntegrator(const ParamSet &params, std::shared_ptr<Sampler> sampler, std::shared_ptr<const Camera> camera);

#endif //PBRT_INTEGRATORS_VPL_H