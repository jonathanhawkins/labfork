import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice Clone Pipeline",
  description: "Record, label, and train your voice model",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
