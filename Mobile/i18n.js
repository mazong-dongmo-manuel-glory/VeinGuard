import i18next from "i18next";
import { initReactI18next } from "react-i18next";
import fr from "./langues/fr.json";
import en from "./langues/en.json";

const initI18next = (langueParDefaut) => {
  if (i18next.isInitialized) {
    i18next.changeLanguage(langueParDefaut || "fr");
    return;
  }

  i18next.use(initReactI18next).init({
    compatibilityJSON: "v4",
    resources: { en: { translation: en }, fr: { translation: fr } },
    lng: langueParDefaut || "fr",
    fallbackLng: "fr",
    interpolation: { escapeValue: false },
  });
};
export default initI18next;
