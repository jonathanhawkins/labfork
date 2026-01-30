/**
 * Loading State for Contribute Page
 *
 * Displayed while the /contribute page is loading.
 * Shows skeleton loaders that match the actual page layout.
 */

import { PageLoadingSkeleton } from "@/components/compute/ComputeLoading";

export default function ContributeLoading() {
  return <PageLoadingSkeleton />;
}
