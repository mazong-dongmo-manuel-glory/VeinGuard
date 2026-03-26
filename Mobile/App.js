import { StatusBar } from "expo-status-bar";
import { StyleSheet, View, Text } from "react-native";
import NavigationRoot from "./navigation/NavigationRoot";
import initI18next from "./i18n";
import { useLangueStore } from "./store/langueStore";
import { useEffect, useState } from "react";
import { useMqttStore } from "./store/mqttStore";
import { useAuthStore } from "./store/authStore";

export default function App() {
  const [pret, setPret] = useState(false);
  const langue = useLangueStore((state) => state.langue);
  const connectMqtt = useMqttStore((state) => state.connect);
  const bootstrapMqtt = useMqttStore((state) => state.bootstrap);
  const mqttReady = useMqttStore((state) => state.configReady);
  const bootstrapAuth = useAuthStore((state) => state.bootstrap);
  const authReady = useAuthStore((state) => state.authReady);

  useEffect(() => {
    initI18next(langue);
    bootstrapMqtt().finally(() => connectMqtt());
    bootstrapAuth().finally(() => setPret(true));
  }, [langue, connectMqtt, bootstrapAuth, bootstrapMqtt]);

  if (!pret || !authReady || !mqttReady) {
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
