import type { Metadata } from "next";
import { IBM_Plex_Mono } from "next/font/google";
import { Providers } from "@/components/providers";
import { Toaster } from "@/components/ui/sonner";
import { Navigation } from "@/components/Navigation";
import { ServiceWorkerRegistration } from "@/components/ServiceWorkerRegistration";
import "./globals.css";

const ibmPlexMono = IBM_Plex_Mono({
  weight: ["400", "500", "600"],
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "LabFork - Open Platform for AI Research",
  description: "Fork research labs, watch AI agents implement papers, discover synergies across domains. The open platform for collective AI research.",
  icons: {
    icon: [
      {
        url: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><defs><linearGradient id='g' x1='0%25' y1='0%25' x2='100%25' y2='100%25'><stop offset='0%25' stop-color='%233b82f6'/><stop offset='100%25' stop-color='%238b5cf6'/></linearGradient></defs><rect fill='url(%23g)' rx='20' width='100' height='100'/><path d='M50 20 L50 45 M35 55 L50 45 L65 55 M35 55 L35 75 M65 55 L65 75' stroke='white' stroke-width='8' stroke-linecap='round' fill='none'/><circle cx='35' cy='78' r='6' fill='white'/><circle cx='65' cy='78' r='6' fill='white'/></svg>",
        type: "image/svg+xml",
      },
    ],
  },
  openGraph: {
    title: "LabFork - Open Platform for AI Research",
    description: "Fork research labs, watch AI agents implement papers, discover synergies across domains.",
    siteName: "LabFork",
    url: "https://labfork.com",
  },
  twitter: {
    card: "summary_large_image",
    title: "LabFork - Open Platform for AI Research",
    description: "Fork research labs, watch AI agents implement papers, discover synergies across domains.",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#3b82f6" />
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="apple-mobile-web-app-title" content="LabFork" />
      </head>
      <body className={`${ibmPlexMono.variable} font-mono`}>
        <Providers>
          <ServiceWorkerRegistration />
          <Navigation />
          {children}
          <Toaster position="bottom-right" />
        </Providers>
      </body>
    </html>
  );
}
