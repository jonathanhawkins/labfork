/**
 * LabFork - Landing Page
 *
 * Main landing page showcasing the platform features,
 * domain showcase, meta-agents, and deployment options.
 */

import {
  HeroSection,
  DomainShowcase,
  HowItWorks,
  MetaAgentFeature,
  ProjectIdeasSection,
  SocialPreview,
  DeploymentOptions,
  LandingFooter,
} from "@/components/landing";

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-slate-950">
      <HeroSection />
      <DomainShowcase />
      <HowItWorks />
      <MetaAgentFeature />
      <ProjectIdeasSection />
      <SocialPreview />
      <DeploymentOptions />
      <LandingFooter />
    </main>
  );
}
