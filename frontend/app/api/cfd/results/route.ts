import { NextResponse } from 'next/server';

const CFD_API_URL = process.env.CFD_API_URL || 'http://100.83.78.111:8007';

export async function GET() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const response = await fetch(`${CFD_API_URL}/results`, {
      signal: controller.signal,
      headers: {
        'Accept': 'application/json',
      },
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`CFD API returned ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('CFD API error:', error);
    return NextResponse.json(
      { error: 'CFD API unavailable', message: String(error) },
      { status: 503 }
    );
  }
}
