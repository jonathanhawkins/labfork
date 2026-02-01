/**
 * i18n Configuration
 *
 * Supported locales and their metadata for LabFork's global reach.
 */

export const locales = ['en', 'es', 'hi', 'zh', 'ar'] as const;
export type Locale = (typeof locales)[number];

export const defaultLocale: Locale = 'en';

export const localeNames: Record<Locale, string> = {
  en: 'English',
  es: 'Español',
  hi: 'हिन्दी',
  zh: '中文',
  ar: 'العربية',
};

export const localeFlags: Record<Locale, string> = {
  en: '🇺🇸',
  es: '🇪🇸',
  hi: '🇮🇳',
  zh: '🇨🇳',
  ar: '🇸🇦',
};

// RTL languages
export const rtlLocales: Locale[] = ['ar'];

export function isRtlLocale(locale: Locale): boolean {
  return rtlLocales.includes(locale);
}
