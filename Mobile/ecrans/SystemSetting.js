import React, { useEffect, useState } from 'react';
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
  Alert,
} from 'react-native';
import { useTranslation } from 'react-i18next';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { COLORS, GRADIENTS } from '../theme';
import { getAppErrorMessage } from '../services/appErrors';
import { useMqttStore } from '../store/mqttStore';
import { useAuthStore } from '../store/authStore';
import { useLangueStore } from '../store/langueStore';

function SectionHeader({ icon, title, color = COLORS.neonCyan }) {
  return (
    <View style={styles.sectionHeader}>
      <Ionicons name={icon} size={20} color={color} />
      <Text style={[styles.sectionTitle, { color }]}>{title}</Text>
    </View>
  );
}

function ToggleRow({ title, description, value, onValueChange, color }) {
  return (
    <View style={styles.settingRow}>
      <View style={styles.settingCopy}>
        <Text style={styles.settingLabel}>{title}</Text>
        <Text style={styles.settingDesc}>{description}</Text>
      </View>
      <Switch value={value} onValueChange={onValueChange} trackColor={{ true: color }} />
    </View>
  );
}

function TelemetryItem({ label, value, accent = COLORS.white }) {
  return (
    <View style={styles.telemetryItem}>
      <Text style={styles.telemetryLabel}>{label}</Text>
      <Text style={[styles.telemetryValue, { color: accent }]}>{String(value)}</Text>
    </View>
  );
}

export default function SystemSetting({ navigation }) {
  const { t } = useTranslation();
  const insets = useSafeAreaInsets();
  const isConnected = useMqttStore((state) => state.isConnected);
  const telemetry = useMqttStore((state) => state.telemetry);
  const updateSystemSettings = useMqttStore((state) => state.updateSystemSettings);
  const brokerConfig = useMqttStore((state) => state.brokerConfig);
  const updateBrokerConfig = useMqttStore((state) => state.updateBrokerConfig);
  const preferences = useAuthStore((state) => state.preferences);
  const updatePreferences = useAuthStore((state) => state.updatePreferences);
  const langue = useLangueStore((state) => state.langue);
  const modifierLangue = useLangueStore((state) => state.modifierLangue);

  const [autoLightEnabled, setAutoLightEnabled] = useState(true);
  const [assistLightsOn, setAssistLightsOn] = useState(false);
  const [greenLedOn, setGreenLedOn] = useState(false);
  const [redLedOn, setRedLedOn] = useState(false);
  const [darkRatio, setDarkRatio] = useState('1.25');
  const [lcdLine1, setLcdLine1] = useState('BioGuard');
  const [lcdLine2, setLcdLine2] = useState('');
  const [serverHost, setServerHost] = useState('');
  const [serverWsPort, setServerWsPort] = useState('');
  const [serverMqttPort, setServerMqttPort] = useState('');

  useEffect(() => {
    if (!telemetry) return;
    setAutoLightEnabled(Boolean(telemetry.lighting?.auto_enabled));
    setAssistLightsOn(Boolean(telemetry.lighting?.assist_lights_on));
    setGreenLedOn(Boolean(telemetry.lighting?.green_led_on));
    setRedLedOn(Boolean(telemetry.lighting?.red_led_on));
    if (telemetry.light_sensor?.dark_ratio != null) {
      setDarkRatio(String(telemetry.light_sensor.dark_ratio));
    }
    if (telemetry.lcd?.line1) {
      setLcdLine1(telemetry.lcd.line1);
    }
    setLcdLine2(telemetry.lcd?.line2 || '');
  }, [telemetry]);

  useEffect(() => {
    setServerHost(brokerConfig.host || '');
    setServerWsPort(String(brokerConfig.wsPort || ''));
    setServerMqttPort(String(brokerConfig.mqttPort || ''));
  }, [brokerConfig]);

  const sendSettings = async (payload, successMessage = t('systemSettings.hardwareCommandSent')) => {
    if (!isConnected) {
      Alert.alert(t('systemSettings.systemOfflineTitle'), t('systemSettings.systemOfflineDesc'));
      return;
    }

    try {
      await updateSystemSettings(payload);
      Alert.alert(t('common.success'), successMessage);
    } catch (error) {
      Alert.alert(t('common.error'), getAppErrorMessage(t, error, 'systemSettings.systemOfflineDesc'));
    }
  };

  const handlePreferenceChange = async (patch) => {
    try {
      await updatePreferences(patch);
    } catch (error) {
      Alert.alert(t('common.error'), getAppErrorMessage(t, error, 'systemSettings.preferencesError'));
    }
  };

  const handleConnectionSave = async () => {
    if (!serverHost.trim() || !serverWsPort.trim() || !serverMqttPort.trim()) {
      Alert.alert(t('common.error'), t('systemSettings.connectionRequired'));
      return;
    }

    try {
      await updateBrokerConfig({
        host: serverHost.trim(),
        wsPort: serverWsPort.trim(),
        mqttPort: serverMqttPort.trim(),
      });
      Alert.alert(t('common.success'), t('systemSettings.connectionSaved'));
    } catch (error) {
      Alert.alert(t('common.error'), getAppErrorMessage(t, error, 'systemSettings.connectionError'));
    }
  };

  const handleImmediateToggle = async (setter, payload, successMessage) => {
    setter(payload[Object.keys(payload)[0]]);
    await sendSettings(payload, successMessage);
  };

  const applyLightingSettings = async () => {
    await sendSettings({
      auto_light_enabled: autoLightEnabled,
      assist_lights_on: assistLightsOn,
      green_led_on: greenLedOn,
      red_led_on: redLedOn,
      dark_ratio: Number.parseFloat(darkRatio) || 1.25,
    });
  };

  const handleSendLcd = async () => {
    await sendSettings(
      {
        lcd_line1: lcdLine1,
        lcd_line2: lcdLine2,
      },
      t('systemSettings.lcdSend'),
    );
  };

  const lightData = telemetry?.light_sensor;
  const lightingData = telemetry?.lighting;
  const lcdData = telemetry?.lcd;
  const cameraData = telemetry?.camera;
  const languageOptions = [
    { value: 'fr', label: t('systemSettings.french') },
    { value: 'en', label: t('systemSettings.english') },
  ];

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <View style={[styles.header, { paddingTop: Math.max(insets.top + 8, 20) }]}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('systemSettings.headerTitle')}</Text>
        <View style={styles.spacer} />
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        <View style={styles.titleSection}>
          <Text style={styles.pageTitle}>{t('systemSettings.hardwareTitle')}</Text>
          <Text style={styles.pageSubtitle}>{t('systemSettings.hardwareSubtitle')}</Text>
        </View>

        <View style={styles.card}>
          <SectionHeader icon="pulse-outline" title={t('systemSettings.telemetryTitle')} color={COLORS.neonGreen} />
          {telemetry ? (
            <View style={styles.telemetryGrid}>
              <TelemetryItem
                label={t('systemSettings.lightValue')}
                value={lightData?.value ?? '--'}
                accent={lightData?.is_dark ? COLORS.neonAmber : COLORS.neonGreen}
              />
              <TelemetryItem label={t('systemSettings.lightBaseline')} value={lightData?.baseline ?? '--'} />
              <TelemetryItem label={t('systemSettings.lightThreshold')} value={lightData?.dark_threshold ?? '--'} />
              <TelemetryItem
                label={t('systemSettings.lightSensorTitle')}
                value={lightData?.is_dark ? t('systemSettings.lightDark') : t('systemSettings.lightBright')}
                accent={lightData?.is_dark ? COLORS.neonAmber : COLORS.neonGreen}
              />
              <TelemetryItem
                label={t('systemSettings.lightingState')}
                value={lightingData?.assist_lights_on ? t('common.online') : t('common.offline')}
                accent={lightingData?.assist_lights_on ? COLORS.neonGreen : COLORS.textDim}
              />
              <TelemetryItem
                label={t('systemSettings.cameraTitle')}
                value={cameraData?.available ? t('systemSettings.cameraAvailable') : t('common.offline')}
                accent={cameraData?.available ? COLORS.neonGreen : COLORS.neonRed}
              />
              <TelemetryItem
                label={t('systemSettings.cameraMockMode')}
                value={cameraData?.mock_mode ? t('common.warning') : t('common.success')}
                accent={cameraData?.mock_mode ? COLORS.neonAmber : COLORS.neonGreen}
              />
              <TelemetryItem
                label={t('systemSettings.greenLedState')}
                value={lightingData?.green_led_on ? t('common.online') : t('common.offline')}
                accent={lightingData?.green_led_on ? COLORS.neonGreen : COLORS.textDim}
              />
              <TelemetryItem
                label={t('systemSettings.redLedState')}
                value={lightingData?.red_led_on ? t('common.online') : t('common.offline')}
                accent={lightingData?.red_led_on ? COLORS.neonRed : COLORS.textDim}
              />
              <TelemetryItem label={t('systemSettings.lcdLine1')} value={lcdData?.line1 || '--'} />
              <TelemetryItem label={t('systemSettings.lcdLine2')} value={lcdData?.line2 || '--'} />
            </View>
          ) : (
            <Text style={styles.emptyText}>{t('systemSettings.telemetryUnavailable')}</Text>
          )}
        </View>

        <View style={styles.card}>
          <SectionHeader icon="phone-portrait-outline" title={t('systemSettings.mobilePreferencesTitle')} color={COLORS.neonCyan} />
          <View style={styles.cardContent}>
            <ToggleRow
              title={t('systemSettings.prefAutoRefreshTitle')}
              description={t('systemSettings.prefAutoRefreshDesc')}
              value={Boolean(preferences?.autoRefreshData)}
              onValueChange={(value) => handlePreferenceChange({ autoRefreshData: value })}
              color={COLORS.neonGreen}
            />
            <ToggleRow
              title={t('systemSettings.prefCompactListsTitle')}
              description={t('systemSettings.prefCompactListsDesc')}
              value={Boolean(preferences?.compactLists)}
              onValueChange={(value) => handlePreferenceChange({ compactLists: value })}
              color={COLORS.neonAmber}
            />
            <ToggleRow
              title={t('systemSettings.prefTechnicalTitle')}
              description={t('systemSettings.prefTechnicalDesc')}
              value={Boolean(preferences?.showTechnicalDetails)}
              onValueChange={(value) => handlePreferenceChange({ showTechnicalDetails: value })}
              color={COLORS.neonCyan}
            />
            <Text style={styles.helperText}>{t('systemSettings.prefPersistenceNote')}</Text>
          </View>
        </View>

        <View style={styles.card}>
          <SectionHeader icon="language-outline" title={t('systemSettings.language')} color={COLORS.neonCyan} />
          <View style={styles.cardContent}>
            <Text style={styles.helperText}>{t('systemSettings.selectLanguage')}</Text>
            <Text style={styles.languageCurrent}>
              {t('systemSettings.currentLanguage')} {langue === 'en' ? t('systemSettings.english') : t('systemSettings.french')}
            </Text>
            <View style={styles.languageOptions}>
              {languageOptions.map((option) => {
                const active = langue === option.value;
                return (
                  <TouchableOpacity
                    key={option.value}
                    style={[
                      styles.languageChip,
                      active && styles.languageChipActive,
                    ]}
                    onPress={() => modifierLangue(option.value)}
                    activeOpacity={0.85}
                  >
                    <Text
                      style={[
                        styles.languageChipText,
                        active && styles.languageChipTextActive,
                      ]}
                    >
                      {option.label}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          </View>
        </View>

        <View style={styles.card}>
          <SectionHeader icon="globe-outline" title={t('systemSettings.connectionTitle')} color={COLORS.neonCyan} />
          <View style={styles.cardContent}>
            <Text style={styles.helperText}>{t('systemSettings.connectionDesc')}</Text>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>{t('systemSettings.connectionHost')}</Text>
              <TextInput style={styles.textInput} value={serverHost} onChangeText={setServerHost} autoCapitalize="none" />
            </View>
            <View style={styles.portRow}>
              <View style={[styles.inputGroup, styles.portGroup]}>
                <Text style={styles.inputLabel}>{t('systemSettings.connectionWsPort')}</Text>
                <TextInput style={styles.textInput} value={serverWsPort} onChangeText={setServerWsPort} keyboardType="number-pad" />
              </View>
              <View style={[styles.inputGroup, styles.portGroup]}>
                <Text style={styles.inputLabel}>{t('systemSettings.connectionTcpPort')}</Text>
                <TextInput style={styles.textInput} value={serverMqttPort} onChangeText={setServerMqttPort} keyboardType="number-pad" />
              </View>
            </View>
            <TouchableOpacity style={styles.primaryBtn} onPress={handleConnectionSave}>
              <Text style={styles.primaryBtnText}>{t('systemSettings.connectionSave')}</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.card}>
          <SectionHeader icon="bulb-outline" title={t('systemSettings.lightSensorTitle')} color={COLORS.neonAmber} />
          <View style={styles.cardContent}>
            <ToggleRow
              title={t('systemSettings.autoLightTitle')}
              description={t('systemSettings.autoLightDesc')}
              value={autoLightEnabled}
              onValueChange={setAutoLightEnabled}
              color={COLORS.neonAmber}
            />
            <ToggleRow
              title={t('systemSettings.assistLightsTitle')}
              description={t('systemSettings.assistLightsDesc')}
              value={assistLightsOn}
              onValueChange={(value) => handleImmediateToggle(setAssistLightsOn, { assist_lights_on: value }, t('systemSettings.hardwareCommandSent'))}
              color={COLORS.neonAmber}
            />
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>{t('systemSettings.lightThreshold')}</Text>
              <TextInput
                style={styles.textInput}
                value={darkRatio}
                onChangeText={setDarkRatio}
                keyboardType="decimal-pad"
              />
            </View>
            <TouchableOpacity style={styles.primaryBtn} onPress={applyLightingSettings}>
              <Text style={styles.primaryBtnText}>{t('systemSettings.updateConfiguration')}</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.card}>
          <SectionHeader icon="color-filter-outline" title={t('systemSettings.securityTitle')} color={COLORS.neonCyan} />
          <View style={styles.cardContent}>
            <ToggleRow
              title={t('systemSettings.greenLedTitle')}
              description={t('systemSettings.greenLedDesc')}
              value={greenLedOn}
              onValueChange={(value) => handleImmediateToggle(setGreenLedOn, { green_led_on: value }, t('systemSettings.hardwareCommandSent'))}
              color={COLORS.neonGreen}
            />
            <ToggleRow
              title={t('systemSettings.redLedTitle')}
              description={t('systemSettings.redLedDesc')}
              value={redLedOn}
              onValueChange={(value) => handleImmediateToggle(setRedLedOn, { red_led_on: value }, t('systemSettings.hardwareCommandSent'))}
              color={COLORS.neonRed}
            />
            <TouchableOpacity
              style={[styles.secondaryBtn, { borderColor: COLORS.neonCyan }]}
              onPress={() => sendSettings({ buzzer_test: true }, t('systemSettings.buzzerButton'))}
            >
              <Ionicons name="volume-high-outline" size={18} color={COLORS.neonCyan} />
              <Text style={[styles.secondaryBtnText, { color: COLORS.neonCyan }]}>{t('systemSettings.buzzerButton')}</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.card}>
          <SectionHeader icon="tablet-portrait-outline" title={t('systemSettings.lcdTitle')} color={COLORS.white} />
          <View style={styles.cardContent}>
            <Text style={styles.helperText}>{t('systemSettings.lcdDesc')}</Text>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>{t('systemSettings.lcdLine1')}</Text>
              <TextInput style={styles.textInput} value={lcdLine1} onChangeText={setLcdLine1} maxLength={16} />
            </View>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>{t('systemSettings.lcdLine2')}</Text>
              <TextInput style={styles.textInput} value={lcdLine2} onChangeText={setLcdLine2} maxLength={16} />
            </View>
            <TouchableOpacity style={styles.primaryBtn} onPress={handleSendLcd}>
              <Text style={styles.primaryBtnText}>{t('systemSettings.lcdSend')}</Text>
            </TouchableOpacity>
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
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  backBtn: { width: 40, height: 40, justifyContent: 'center' },
  headerTitle: { color: COLORS.white, fontSize: 16, fontWeight: '800', letterSpacing: 2 },
  spacer: { width: 40 },
  scroll: { flex: 1, paddingHorizontal: 20 },
  scrollContent: { paddingTop: 10 },
  titleSection: { marginBottom: 30 },
  pageTitle: { color: COLORS.white, fontSize: 30, fontWeight: '900', letterSpacing: 1 },
  pageSubtitle: { color: COLORS.textDim, fontSize: 11, fontWeight: '700', lineHeight: 18, marginTop: 8 },
  card: { marginBottom: 24 },
  cardContent: { gap: 14 },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 14 },
  sectionTitle: { fontSize: 12, fontWeight: '900', letterSpacing: 1.5 },
  telemetryGrid: {
    gap: 12,
  },
  telemetryItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.04)',
    gap: 12,
  },
  telemetryLabel: { color: COLORS.textSecondary, fontSize: 12, fontWeight: '700', flex: 1 },
  telemetryValue: { fontSize: 12, fontWeight: '900', textAlign: 'right', flexShrink: 1 },
  emptyText: { color: COLORS.textDim, fontSize: 13, lineHeight: 20 },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.04)',
    gap: 12,
  },
  settingCopy: { flex: 1, minWidth: 0 },
  settingLabel: { color: COLORS.white, fontSize: 14, fontWeight: '800' },
  settingDesc: { color: COLORS.textDim, fontSize: 11, lineHeight: 16, marginTop: 4 },
  inputGroup: { gap: 8 },
  portRow: { flexDirection: 'row', gap: 12 },
  portGroup: { flex: 1 },
  inputLabel: { color: COLORS.textDim, fontSize: 10, fontWeight: '900', letterSpacing: 1 },
  textInput: {
    height: 50,
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.06)',
    color: COLORS.white,
    paddingHorizontal: 14,
    fontSize: 14,
    fontWeight: '600',
  },
  helperText: { color: COLORS.textSecondary, fontSize: 12, lineHeight: 18 },
  languageCurrent: { color: COLORS.white, fontSize: 12, fontWeight: '700' },
  languageOptions: { flexDirection: 'row', gap: 12, flexWrap: 'wrap' },
  languageChip: {
    minHeight: 44,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    backgroundColor: 'rgba(255,255,255,0.03)',
    paddingHorizontal: 16,
    paddingVertical: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  languageChipActive: {
    borderColor: COLORS.neonCyan,
    backgroundColor: 'rgba(0,229,255,0.16)',
  },
  languageChipText: {
    color: COLORS.textSecondary,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 1,
  },
  languageChipTextActive: {
    color: COLORS.neonCyan,
  },
  primaryBtn: {
    height: 50,
    borderRadius: 12,
    backgroundColor: COLORS.neonCyan,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 4,
  },
  primaryBtnText: { color: COLORS.bg, fontSize: 12, fontWeight: '900', letterSpacing: 1 },
  secondaryBtn: {
    minHeight: 50,
    borderRadius: 12,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 10,
    paddingHorizontal: 14,
    paddingVertical: 14,
  },
  secondaryBtnText: { fontSize: 12, fontWeight: '900', letterSpacing: 1, textAlign: 'center' },
});
