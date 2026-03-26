import React, { useEffect, useState } from "react";
import { StatusBar } from "expo-status-bar";
import { StyleSheet, View, Text } from "react-native";
import NavigationRoot from "./navigation/NavigationRoot";
import initI18next from "./i18n";
import { useLangueStore } from "./store/langueStore";
import { useMqttStore } from "./store/mqttStore";
import { useAuthStore } from "./store/authStore";
import i18next from "i18next";

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error) {
    console.error("App boundary intercepted:", error);
  }

  handleRetry = () => {
    this.setState({ hasError: false });
  };

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <View style={styles.splashContainer}>
        <Text style={styles.boundaryTitle}>{i18next.t("common.appRecoveredTitle")}</Text>
        <Text style={styles.boundaryText}>{i18next.t("common.appRecoveredDesc")}</Text>
        <Text style={styles.boundaryRetry} onPress={this.handleRetry}>
          {i18next.t("common.retry")}
        </Text>
      </View>
    );
  }
}

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
    bootstrapMqtt()
      .finally(() => Promise.resolve(connectMqtt()).catch(() => {}));
    bootstrapAuth().finally(() => setPret(true));
  }, [langue, connectMqtt, bootstrapAuth, bootstrapMqtt]);

  if (!pret || !authReady || !mqttReady) {
    return (
      <View style={styles.splashContainer}>
        <Text style={styles.loadingText}>{i18next.t("common.loading")}</Text>
      </View>
    );
  }

  return (
    <AppErrorBoundary>
      <View style={styles.container}>
        <NavigationRoot />
        <StatusBar style="auto" />
      </View>
    </AppErrorBoundary>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  splashContainer: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
  loadingText: {
    fontSize: 16,
    fontWeight: "700",
  },
  boundaryTitle: {
    fontSize: 20,
    fontWeight: "800",
    marginBottom: 12,
  },
  boundaryText: {
    fontSize: 14,
    lineHeight: 20,
    textAlign: "center",
    paddingHorizontal: 24,
    marginBottom: 16,
  },
  boundaryRetry: {
    fontSize: 15,
    fontWeight: "800",
  },
});
