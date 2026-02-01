/**
 * HowItWorks
 *
 * Three-step process explanation with auto-advancing carousel.
 * Fully internationalized for global accessibility.
 */

"use client";

import React, { useState, useEffect } from "react";
import { useTranslations } from "next-intl";

interface Step {
  number: number;
  titleKey: string;
  descKey: string;
  icon: React.ReactNode;
  color: string;
  detailKeys: string[];
}

const steps: Step[] = [
  {
    number: 1,
    titleKey: "step1Title",
    descKey: "step1Desc",
    icon: (
      <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
      </svg>
    ),
    color: "#3b82f6",
    detailKeys: ["step1Detail1", "step1Detail2", "step1Detail3", "step1Detail4"],
  },
  {
    number: 2,
    titleKey: "step2Title",
    descKey: "step2Desc",
    icon: (
      <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    color: "#8b5cf6",
    detailKeys: ["step2Detail1", "step2Detail2", "step2Detail3", "step2Detail4"],
  },
  {
    number: 3,
    titleKey: "step3Title",
    descKey: "step3Desc",
    icon: (
      <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
    color: "#10b981",
    detailKeys: ["step3Detail1", "step3Detail2", "step3Detail3", "step3Detail4"],
  },
];

export function HowItWorks() {
  const t = useTranslations("howItWorks");
  const [activeStep, setActiveStep] = useState(0);
  const [isPaused, setIsPaused] = useState(false);

  // Auto-advance carousel
  useEffect(() => {
    if (isPaused) return;

    const interval = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % steps.length);
    }, 5000);

    return () => clearInterval(interval);
  }, [isPaused]);

  return (
    <section className="py-24 bg-gradient-to-b from-slate-950 to-slate-900">
      <div className="max-w-6xl mx-auto px-6">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            {t("title")}
          </h2>
          <p className="text-xl text-gray-400">
            {t("subtitle")}
          </p>
        </div>

        {/* Steps Navigation */}
        <div className="flex items-center justify-center gap-4 mb-12">
          {steps.map((step, idx) => (
            <button
              key={step.number}
              onClick={() => {
                setActiveStep(idx);
                setIsPaused(true);
              }}
              className={`flex items-center gap-3 px-6 py-3 min-h-[44px] rounded-xl transition-all active:scale-[0.98] ${
                activeStep === idx
                  ? "bg-white/10 border border-white/20"
                  : "bg-transparent border border-transparent hover:bg-white/5 active:bg-white/10"
              }`}
              aria-label={t("goToStep", { number: step.number })}
            >
              <span
                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-colors ${
                  activeStep === idx
                    ? "text-white"
                    : "text-gray-500"
                }`}
                style={{
                  backgroundColor: activeStep === idx ? step.color : "transparent",
                  border: activeStep === idx ? "none" : "1px solid #4b5563",
                }}
              >
                {step.number}
              </span>
              <span
                className={`hidden sm:block font-medium transition-colors ${
                  activeStep === idx ? "text-white" : "text-gray-500"
                }`}
              >
                {t(step.titleKey)}
              </span>
            </button>
          ))}
        </div>

        {/* Progress Bar */}
        <div className="max-w-2xl mx-auto mb-12">
          <div className="h-1 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full transition-all duration-300"
              style={{
                width: `${((activeStep + 1) / steps.length) * 100}%`,
                backgroundColor: steps[activeStep].color,
              }}
            />
          </div>
        </div>

        {/* Active Step Content */}
        <div
          className="relative rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm overflow-hidden"
          onMouseEnter={() => setIsPaused(true)}
          onMouseLeave={() => setIsPaused(false)}
          onTouchStart={() => setIsPaused(true)}
        >
          {/* Color accent */}
          <div
            className="absolute top-0 left-0 w-full h-1"
            style={{ backgroundColor: steps[activeStep].color }}
          />

          <div className="p-8 md:p-12">
            <div className="grid md:grid-cols-2 gap-12 items-center">
              {/* Left: Icon & Title */}
              <div>
                <div
                  className="inline-flex p-4 rounded-2xl mb-6"
                  style={{ backgroundColor: `${steps[activeStep].color}20` }}
                >
                  <div style={{ color: steps[activeStep].color }}>
                    {steps[activeStep].icon}
                  </div>
                </div>

                <h3 className="text-2xl md:text-3xl font-bold text-white mb-4">
                  {t(steps[activeStep].titleKey)}
                </h3>

                <p className="text-lg text-gray-400 mb-6">
                  {t(steps[activeStep].descKey)}
                </p>

                {/* Step indicator dots - with 44px touch targets */}
                <div className="flex items-center gap-1">
                  {steps.map((_, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setActiveStep(idx);
                        setIsPaused(true);
                      }}
                      className="min-w-[44px] min-h-[44px] flex items-center justify-center active:scale-95"
                      aria-label={t("goToStep", { number: idx + 1 })}
                    >
                      <span
                        className={`h-2 rounded-full transition-all ${
                          activeStep === idx
                            ? "w-8"
                            : "w-2 bg-gray-600 hover:bg-gray-500"
                        }`}
                        style={{
                          backgroundColor: activeStep === idx ? steps[activeStep].color : undefined,
                        }}
                      />
                    </button>
                  ))}
                  {/* Play/Pause button for mobile */}
                  <button
                    onClick={() => setIsPaused(!isPaused)}
                    className="min-w-[44px] min-h-[44px] flex items-center justify-center ms-2 text-gray-500 hover:text-white active:scale-95"
                    aria-label={isPaused ? t("resumeAutoAdvance") : t("pauseAutoAdvance")}
                  >
                    {isPaused ? (
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    ) : (
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>

              {/* Right: Details */}
              <div className="space-y-4">
                {steps[activeStep].detailKeys.map((detailKey, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-3 p-4 rounded-xl bg-white/5"
                    style={{
                      animationDelay: `${idx * 100}ms`,
                    }}
                  >
                    <div
                      className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5"
                      style={{ backgroundColor: `${steps[activeStep].color}30` }}
                    >
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke={steps[activeStep].color}
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    </div>
                    <span className="text-gray-300">{t(detailKey)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default HowItWorks;
