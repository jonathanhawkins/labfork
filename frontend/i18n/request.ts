import { getRequestConfig } from 'next-intl/server';
import { cookies, headers } from 'next/headers';
import { defaultLocale, locales, type Locale } from './config';

export default getRequestConfig(async () => {
  // Try to get locale from cookie first
  const cookieStore = await cookies();
  const localeCookie = cookieStore.get('NEXT_LOCALE')?.value as Locale | undefined;

  let locale: Locale = defaultLocale;

  if (localeCookie && locales.includes(localeCookie)) {
    locale = localeCookie;
  } else {
    // Fall back to Accept-Language header
    const headerStore = await headers();
    const acceptLanguage = headerStore.get('accept-language') || '';

    // Parse accept-language header
    const browserLocales = acceptLanguage
      .split(',')
      .map((l) => l.split(';')[0].trim().substring(0, 2));

    // Find first matching locale
    for (const browserLocale of browserLocales) {
      if (locales.includes(browserLocale as Locale)) {
        locale = browserLocale as Locale;
        break;
      }
    }
  }

  return {
    locale,
    messages: (await import(`../messages/${locale}.json`)).default,
  };
});
