import { StatusBar } from "expo-status-bar";
import { StyleSheet, View, Text } from "react-native";
import NavigationRoot from "./navigation/NavigationRoot";
import initI18next from "./i18n";
import { useLangueStore } from "./store/langueStore";
import { useEffect, useState } from "react";

export default function App() {
  const [pret, setPret] = useState(false);
  const langue = useLangueStore((state) => state.langue);

  useEffect(() => {
    initI18next(langue);
    setPret(true);
  }, [langue]);

  if (!pret) {
    return (
      <View style={styles.container}>
        <Text>SPLASH SCREEN</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <NavigationRoot />
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
});
