import type { Metadata, ResolvingMetadata } from "next";
import { getLabBySlug } from "@/lib/labs/repository";

/**
 * Generate metadata for lab pages (social sharing)
 */
interface LayoutProps {
  children: React.ReactNode;
  params: Promise<{
    username: string;
    slug: string;
  }>;
}

export async function generateMetadata(
  { params }: LayoutProps,
  parent: ResolvingMetadata
): Promise<Metadata> {
  const { username, slug } = await params;

  // Fetch lab data
  const lab = await getLabBySlug(username, slug);

  if (!lab) {
    return {
      title: "Lab Not Found",
      description: "The requested lab could not be found.",
    };
  }

  const title = `${lab.name} - ${lab.owner.displayName}`;
  const description = lab.description || `${lab.domainName} research lab by ${lab.owner.displayName}`;
  const url = `https://voiceclone.lab/labs/${username}/${slug}`;

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      url,
      siteName: "Voice Clone Lab",
      type: "website",
      images: [
        {
          url: `/api/labs/${lab.id}/og-image`,
          width: 1200,
          height: 630,
          alt: lab.name,
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [`/api/labs/${lab.id}/og-image`],
    },
    alternates: {
      canonical: url,
    },
    other: {
      "lab:id": lab.id,
      "lab:owner": lab.owner.username,
      "lab:domain": lab.domainSlug,
      "lab:stars": String(lab.stats.stars),
      "lab:forks": String(lab.stats.forks),
    },
  };
}

export default function LabLayout({ children }: LayoutProps) {
  return children;
}
