import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import type { LabConfig } from "@/lib/lab-wizard/types";
import {
  generateDomainYamlFromConfig,
  generateInitialTasksFromConfig,
  validateLabConfig,
  generatePrompt,
} from "@/lib/lab-wizard/scaffolding";

/**
 * Lab Creation API
 *
 * POST /api/lab/create - Create a new lab from wizard config
 */
export async function POST(request: NextRequest) {
  try {
    const config: LabConfig = await request.json();

    // Validate config
    const validation = validateLabConfig(config);
    if (!validation.valid) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid configuration",
          validationErrors: validation.errors,
        },
        { status: 400 }
      );
    }

    const createdFiles: string[] = [];
    const createdTaskIds: string[] = [];
    let domainSlug: string;

    // Determine domain slug
    if (config.createNewDomain && config.domain?.slug) {
      domainSlug = config.domain.slug;
    } else if (config.existingDomainSlug) {
      domainSlug = config.existingDomainSlug;
    } else {
      return NextResponse.json(
        { success: false, error: "No domain specified" },
        { status: 400 }
      );
    }

    // Create domain if new
    if (config.createNewDomain && config.domain) {
      try {
        // Generate domain.yaml
        const { yaml, config: domainConfig } = generateDomainYamlFromConfig(config);

        // Get project root (parent of frontend)
        const projectRoot = path.resolve(process.cwd(), "..");
        const domainDir = path.join(projectRoot, ".domains", domainSlug);

        // Create directory structure
        await fs.mkdir(domainDir, { recursive: true });
        await fs.mkdir(path.join(domainDir, "prompts"), { recursive: true });
        await fs.mkdir(path.join(domainDir, "data"), { recursive: true });
        await fs.mkdir(path.join(domainDir, "checkpoints"), { recursive: true });
        await fs.mkdir(path.join(domainDir, "outputs"), { recursive: true });

        // Write domain.yaml
        const yamlPath = path.join(domainDir, "domain.yaml");
        await fs.writeFile(yamlPath, yaml, "utf-8");
        createdFiles.push(`.domains/${domainSlug}/domain.yaml`);

        // Write prompt templates
        const researchPrompt = generatePrompt("research", domainSlug, config.domain);
        await fs.writeFile(
          path.join(domainDir, "prompts", "research.md"),
          researchPrompt,
          "utf-8"
        );
        createdFiles.push(`.domains/${domainSlug}/prompts/research.md`);

        const implPrompt = generatePrompt("implementation", domainSlug, config.domain);
        await fs.writeFile(
          path.join(domainDir, "prompts", "implementation.md"),
          implPrompt,
          "utf-8"
        );
        createdFiles.push(`.domains/${domainSlug}/prompts/implementation.md`);

        const evalPrompt = generatePrompt("evaluation", domainSlug, config.domain);
        await fs.writeFile(
          path.join(domainDir, "prompts", "evaluation.md"),
          evalPrompt,
          "utf-8"
        );
        createdFiles.push(`.domains/${domainSlug}/prompts/evaluation.md`);

        // Save domain to domains API as well
        try {
          const domainsResponse = await fetch(
            `${process.env.NEXT_PUBLIC_BASE_URL || "http://localhost:3003"}/api/domains`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(domainConfig),
            }
          );

          if (!domainsResponse.ok) {
            console.warn("Failed to save domain to API, but files were created");
          }
        } catch (apiError) {
          console.warn("Could not update domains API:", apiError);
        }
      } catch (fsError) {
        console.error("Failed to create domain files:", fsError);
        // Continue anyway - files might be created in a different location
      }
    }

    // Generate initial tasks
    const initialTasks = generateInitialTasksFromConfig(config);

    // Create tasks via API
    try {
      for (const task of initialTasks) {
        const taskResponse = await fetch(
          `${process.env.NEXT_PUBLIC_BASE_URL || "http://localhost:3003"}/api/tasks`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              subject: task.subject,
              description: task.description,
              metadata: {
                domainSlug,
                type: task.type,
                estimatedHours: task.estimatedHours,
                priority: task.priority,
                createdBy: "lab-wizard",
              },
            }),
          }
        );

        if (taskResponse.ok) {
          const taskData = await taskResponse.json();
          if (taskData.id) {
            createdTaskIds.push(taskData.id);
          }
        }
      }
    } catch (taskError) {
      console.warn("Failed to create some tasks:", taskError);
      // Continue - lab is still created even if tasks fail
    }

    // Generate lab ID
    const labId = `lab_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;

    // Save lab config (for future reference)
    try {
      const labConfigPath = path.join(
        process.cwd(),
        "data",
        "labs",
        `${labId}.json`
      );
      await fs.mkdir(path.dirname(labConfigPath), { recursive: true });
      await fs.writeFile(
        labConfigPath,
        JSON.stringify(
          {
            id: labId,
            domainSlug,
            config,
            createdFiles,
            createdTaskIds,
            createdAt: new Date().toISOString(),
          },
          null,
          2
        ),
        "utf-8"
      );
    } catch (saveError) {
      console.warn("Failed to save lab config:", saveError);
    }

    return NextResponse.json({
      success: true,
      labId,
      domainSlug,
      files: createdFiles,
      taskIds: createdTaskIds,
      redirectUrl: `/lab?domain=${domainSlug}`,
    });
  } catch (error) {
    console.error("Lab creation error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Lab creation failed",
      },
      { status: 500 }
    );
  }
}
