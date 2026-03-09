import type { Metadata } from "next";
import { IBM_Plex_Mono } from "next/font/google";
import { NextIntlClientProvider } from "next-intl";
import { getLocale, getMessages } from "next-intl/server";
import { Providers } from "@/components/providers";
import { Toaster } from "@/components/ui/sonner";
import { Navigation } from "@/components/Navigation";
import { ServiceWorkerRegistration } from "@/components/ServiceWorkerRegistration";
import { isRtlLocale, type Locale } from "@/i18n/config";
import "./globals.css";

const ibmPlexMono = IBM_Plex_Mono({
  weight: ["400", "500", "600"],
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "LabFork - Open Platform for AI Research",
  description:
    "Fork research labs, watch AI agents implement papers, discover synergies across domains. The open platform for collective AI research.",
  metadataBase: new URL("https://labfork.com"),
  openGraph: {
    title: "LabFork - Open Platform for AI Research",
    description:
      "Fork research labs, watch AI agents implement papers, discover synergies across domains.",
    siteName: "LabFork",
    url: "https://labfork.com",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "LabFork - Open Platform for AI Research",
    description:
      "Fork research labs, watch AI agents implement papers, discover synergies across domains.",
    site: "@labfork",
    creator: "@labfork",
  },
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const locale = await getLocale() as Locale;
  const messages = await getMessages();
  const isRtl = isRtlLocale(locale);

  return (
    <html
      lang={locale}
      dir={isRtl ? "rtl" : "ltr"}
      className="dark"
      suppressHydrationWarning
    >
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#3b82f6" />
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="apple-mobile-web-app-title" content="LabFork" />
      </head>
      <body className={`${ibmPlexMono.variable} font-mono`}>
        <NextIntlClientProvider messages={messages}>
          <Providers>
            <ServiceWorkerRegistration />
            <Navigation />
            {children}
            <Toaster position={isRtl ? "bottom-left" : "bottom-right"} />
          </Providers>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
