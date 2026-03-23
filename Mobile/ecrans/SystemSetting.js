import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Switch,
  StatusBar,
  Platform,
} from 'react-native';
import { useTranslation } from 'react-i18next';
import { useLangueStore } from '../store/langueStore';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { COLORS, GRADIENTS } from '../theme';

function SectionHeader({ icon, title, color = COLORS.neonCyan }) {
  return (
    <View style={styles.sectionHeader}>
      <Ionicons name={icon} size={20} color={color} />
      <Text style={[styles.sectionTitle, { color }]}>{title.toUpperCase()}</Text>
    </View>
  );
}

function InputField({ label, value, onChangeText, placeholder, secure }) {
  return (
    <View style={styles.inputGroup}>
      <Text style={styles.inputLabel}>{label}</Text>
      <TextInput
        style={styles.textInput}
        value={value}
        onChangeText={onChangeText}
        placeholder={placeholder}
        placeholderTextColor="rgba(255, 255, 255, 0.2)"
        secureTextEntry={secure}
      />
    </View>
  );
}

function DeviceCard({ name, status, firmware, uptime, signal }) {
  const isOnline = status === 'Online';
  const color = isOnline ? COLORS.neonGreen : COLORS.neonAmber;
  return (
    <BlurView intensity={10} tint="dark" style={styles.deviceCard}>
      <View style={styles.deviceHeader}>
        <View style={styles.deviceInfo}>
          <View style={[styles.statusDot, { backgroundColor: color, shadowColor: color }]} />
          <Text style={styles.deviceName}>{name}</Text>
        </View>
        <View style={styles.deviceTag}>
          <Text style={[styles.deviceTagText, { color }]}>{status.toUpperCase()}</Text>
        </View>
      </View>
      <View style={styles.deviceGrid}>
        <View style={styles.gridItem}>
          <Text style={styles.gridLabel}>FW</Text>
          <Text style={styles.gridVal}>{firmware}</Text>
        </View>
        <View style={styles.gridItem}>
          <Text style={styles.gridLabel}>UPTIME</Text>
          <Text style={styles.gridVal}>{uptime}</Text>
        </View>
        <View style={styles.gridItem}>
          <Text style={styles.gridLabel}>RSSI</Text>
          <Text style={styles.gridVal}>{signal}</Text>
        </View>
      </View>
      <View style={styles.deviceActions}>
        <TouchableOpacity style={styles.miniBtn}><Text style={styles.miniBtnText}>DIAG</Text></TouchableOpacity>
        <TouchableOpacity style={styles.miniBtn}><Text style={styles.miniBtnText}>REBOOT</Text></TouchableOpacity>
      </View>
    </BlurView>
  );
}

import { useMqttStore } from '../store/mqttStore';
import { Alert } from 'react-native';

export default function SystemSetting({ navigation }) {
  const { t } = useTranslation();
  const langue = useLangueStore((state) => state.langue);
  const modifierLangue = useLangueStore((state) => state.modifierLangue);
  
  const [broker, setBroker] = useState('172.16.9.115');
  const [tls, setTls] = useState(true);
  const [biometric, setBiometric] = useState(true);

  const isConnected = useMqttStore((state) => state.isConnected);
  const client = useMqttStore((state) => state.client);

  const handleUpdateConfig = () => {
    if (!isConnected) {
      Alert.alert("System Offline", "Unable to reach the security gateway.");
      return;
    }

    const config = {
      broker_host: broker,
      tls_enabled: tls,
      biometric_override: biometric,
      timestamp: Date.now()
    };

    client.publish('veinguard/cmd/settings/update', JSON.stringify(config));
    Alert.alert("Configuration Transmitted", "System core settings have been updated across the cluster.");
  };

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>SYSTEM CORE</Text>
        <View style={styles.spacer} />
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        <View style={styles.titleSection}>
          <Text style={styles.pageTitle}>SETTINGS</Text>
          <Text style={styles.pageSubtitle}>CORE INFRASTRUCTURE & SECURITY</Text>
        </View>

        <View style={styles.card}>
          <SectionHeader icon="radio-outline" title="MQTT Broker" color={COLORS.neonCyan} />
          <View style={styles.cardContent}>
            <InputField label="HOST ADDRESS" value={broker} onChangeText={setBroker} />
            <View style={styles.row}>
              <InputField label="PORT" value="1883" onChangeText={() => {}} />
              <View style={styles.toggleGroup}>
                <Text style={styles.inputLabel}>TLS 1.3</Text>
                <Switch value={tls} onValueChange={setTls} trackColor={{ true: COLORS.neonCyan }} />
              </View>
            </View>
            <TouchableOpacity style={styles.primaryBtn} onPress={handleUpdateConfig}>
              <Text style={styles.primaryBtnText}>UPDATE CONFIGURATION</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.card}>
          <SectionHeader icon="hardware-chip-outline" title="Active Nodes" color={COLORS.neonAmber} />
          <View style={styles.cardContent}>
            <DeviceCard name="VEIN-ESP-01" status="Online" firmware="v3.1.2" uptime="12d 4h" signal="-65dBm" />
            <DeviceCard name="VEIN-ESP-02" status="Warning" firmware="v3.0.1" uptime="45m" signal="-82dBm" />
            <TouchableOpacity style={styles.addBtn}>
              <Ionicons name="add" size={20} color={COLORS.neonAmber} />
              <Text style={styles.addBtnText}>PROVISION NEW NODE</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.card}>
          <SectionHeader icon="shield-checkmark-outline" title="Security" color={COLORS.neonGreen} />
          <View style={styles.cardContent}>
            <View style={styles.settingRow}>
              <View>
                <Text style={styles.settingLabel}>BIOMETRIC OVERRIDE</Text>
                <Text style={styles.settingDesc}>Allow admin access with system bio</Text>
              </View>
              <Switch value={biometric} onValueChange={setBiometric} trackColor={{ true: COLORS.neonGreen }} />
            </View>
            <View style={styles.settingRow}>
              <View>
                <Text style={styles.settingLabel}>AUTO-REFRESH LOGS</Text>
                <Text style={styles.settingDesc}>Real-time telemetry streaming</Text>
              </View>
              <Switch value={true} onValueChange={() => {}} trackColor={{ true: COLORS.neonGreen }} />
            </View>
          </View>
        </View>

        <View style={styles.card}>
          <SectionHeader icon="globe-outline" title="Localization" color={COLORS.white} />
          <View style={styles.cardContent}>
            <View style={styles.langRow}>
              <TouchableOpacity 
                style={[styles.langBtn, langue === 'en' && styles.langBtnActive]}
                onPress={() => modifierLangue('en')}
              >
                <Text style={[styles.langBtnText, langue === 'en' && styles.langBtnTextActive]}>ENGLISH</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={[styles.langBtn, langue === 'fr' && styles.langBtnActive]}
                onPress={() => modifierLangue('fr')}
              >
                <Text style={[styles.langBtnText, langue === 'fr' && styles.langBtnTextActive]}>FRANÇAIS</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>

        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: COLORS.bg },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 50 : 20,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  backBtn: { width: 40, height: 40, justifyContent: 'center' },
  headerTitle: { color: COLORS.white, fontSize: 16, fontWeight: '800', letterSpacing: 2 },
  spacer: { width: 40 },

  scroll: { flex: 1, paddingHorizontal: 20 },
  scrollContent: { paddingTop: 10 },

  titleSection: { marginBottom: 30 },
  pageTitle: { color: COLORS.white, fontSize: 32, fontWeight: '900', letterSpacing: 2 },
  pageSubtitle: { color: COLORS.textDim, fontSize: 10, fontWeight: '800', letterSpacing: 1, marginTop: 5 },

  card: { marginBottom: 25 },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 15 },
  sectionTitle: { fontSize: 12, fontWeight: '900', letterSpacing: 2 },
  cardContent: { gap: 15 },

  row: { flexDirection: 'row', gap: 15, alignItems: 'flex-end' },
  inputGroup: { flex: 1 },
  inputLabel: { color: COLORS.textDim, fontSize: 8, fontWeight: '900', letterSpacing: 1, marginBottom: 8 },
  textInput: { 
    height: 50, backgroundColor: 'rgba(255, 255, 255, 0.03)', 
    borderRadius: 12, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)',
    paddingHorizontal: 15, color: COLORS.white, fontSize: 14, fontWeight: '600',
  },
  toggleGroup: { alignItems: 'center', paddingBottom: 5 },

  primaryBtn: { 
    height: 50, borderRadius: 12, backgroundColor: COLORS.neonCyan, 
    alignItems: 'center', justifyContent: 'center', marginTop: 5,
    shadowColor: COLORS.neonCyan, shadowOpacity: 0.3, shadowRadius: 10,
  },
  primaryBtnText: { color: COLORS.bg, fontSize: 12, fontWeight: '900', letterSpacing: 1 },

  settingRow: { 
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingVertical: 15, borderBottomWidth: 1, borderBottomColor: 'rgba(255, 255, 255, 0.03)',
  },
  settingLabel: { color: COLORS.white, fontSize: 14, fontWeight: '800' },
  settingDesc: { color: COLORS.textDim, fontSize: 10, marginTop: 4 },

  deviceCard: { borderRadius: 20, padding: 15, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)', marginBottom: 10, overflow: 'hidden' },
  deviceHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 },
  deviceInfo: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  statusDot: { width: 8, height: 8, borderRadius: 4, shadowOpacity: 1, shadowRadius: 5 },
  deviceName: { color: COLORS.white, fontSize: 14, fontWeight: '800' },
  deviceTag: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6, backgroundColor: 'rgba(255, 255, 255, 0.03)' },
  deviceTagText: { fontSize: 8, fontWeight: '900', letterSpacing: 1 },
  deviceGrid: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 15 },
  gridItem: { alignItems: 'center' },
  gridLabel: { color: COLORS.textDim, fontSize: 7, fontWeight: '800', marginBottom: 4 },
  gridVal: { color: COLORS.white, fontSize: 10, fontWeight: '700' },
  deviceActions: { flexDirection: 'row', gap: 10 },
  miniBtn: { flex: 1, height: 35, borderRadius: 8, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.1)', alignItems: 'center', justifyContent: 'center' },
  miniBtnText: { color: COLORS.textDim, fontSize: 8, fontWeight: '900' },

  addBtn: { 
    height: 50, borderRadius: 12, borderWidth: 1, borderStyle: 'dashed', 
    borderColor: COLORS.neonAmber, alignItems: 'center', justifyContent: 'center',
    flexDirection: 'row', gap: 10,
  },
  addBtnText: { color: COLORS.neonAmber, fontSize: 10, fontWeight: '900', letterSpacing: 1 },

  langRow: { flexDirection: 'row', gap: 15 },
  langBtn: { flex: 1, height: 50, borderRadius: 12, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.1)', alignItems: 'center', justifyContent: 'center' },
  langBtnActive: { borderColor: COLORS.white, backgroundColor: 'rgba(255, 255, 255, 0.05)' },
  langBtnText: { color: COLORS.textDim, fontSize: 11, fontWeight: '800' },
  langBtnTextActive: { color: COLORS.white },
});
