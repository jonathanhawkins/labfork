/**
 * Watch Route Group Layout
 *
 * This layout is specifically for the /watch page which has its own
 * mobile-optimized navigation and doesn't need the global Navigation bar.
 * Using a route group (watch) allows us to have a different layout.
 */
export default function WatchLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Just pass through children - the global layout handles html/body,
  // and the watch page has its own header and bottom navigation
  return <>{children}</>;
}
