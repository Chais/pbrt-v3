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
#include "integrator.h"
#include "paramset.h"
#include "directlighting.h"

STAT_COUNTER("Scene/VirtualLights created", nVirtualLights);

struct VirtualLight {
    VirtualLight() {}

    VirtualLight(const Point3f &pp, const SurfaceInteraction &c, const Spectrum &a) : p(pp), contrib(c), alpha(a) {}

    Point3f p;
    SurfaceInteraction contrib;
    Spectrum alpha;
};

class RichVPLIntegrator : public SamplerIntegrator {
public:
    RichVPLIntegrator(const std::shared_ptr<const Camera> &camera, const std::shared_ptr<Sampler> &sampler,
                      const uint32_t &nl, const uint32_t &ns, const float &gl, const int &ng, const float &rrt,
                      const int &maxd, const LightStrategy &strat, const bool &vl, const bool &dl)
            : SamplerIntegrator(camera, sampler),
              nLightPaths(RoundUpPow2(int32_t(nl))), nLightSets(RoundUpPow2(int32_t(ns))), gLimit(gl),
              nGatherSamples(ng), rrThreshold(rrt), maxDepth(maxd), strategy(strat), showVLights(vl),
              noDirectLighting(dl) {
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
    MemoryArena localArena;
    std::vector<std::vector<VirtualLight>> virtualLights;
    uint32_t vlSetOffset;
    std::vector<int> nLightSamples;
};

RichVPLIntegrator *
CreateRVPLIntegrator(const ParamSet &params, std::shared_ptr<Sampler> sampler, std::shared_ptr<const Camera> camera);

#endif //PBRT_INTEGRATORS_RVPL_H