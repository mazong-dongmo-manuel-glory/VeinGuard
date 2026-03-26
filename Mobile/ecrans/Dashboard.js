import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
  Platform,
  Alert,
  useWindowDimensions,
} from 'react-native';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { COLORS, GRADIENTS } from '../theme';
import { useAuthStore } from '../store/authStore';
import { useMqttStore } from '../store/mqttStore';

function SummaryItem({ label, value, accent = COLORS.white }) {
  return (
    <View style={styles.summaryItem}>
      <Text style={styles.summaryLabel}>{label}</Text>
      <Text style={[styles.summaryValue, { color: accent }]}>{value}</Text>
    </View>
  );
}

export default function Dashboard({ navigation }) {
  const { t } = useTranslation();
  const { width } = useWindowDimensions();
  const [time, setTime] = useState(new Date());
  const isCompact = width < 390;
  
  const systemStatus = useMqttStore((state) => state.status);
  const isConnected = useMqttStore((state) => state.isConnected);
  const telemetry = useMqttStore((state) => state.telemetry);
  const statusPayload = useMqttStore((state) => state.statusPayload);
  const logout = useAuthStore((state) => state.logout);
  const systemStatusLabel =
    systemStatus === 'ONLINE'
      ? t('dashboard.onlineStatus')
      : systemStatus === 'OFFLINE'
        ? t('dashboard.offlineStatus')
        : systemStatus;

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (d) => {
    const h = String(d.getHours()).padStart(2, '0');
    const m = String(d.getMinutes()).padStart(2, '0');
    return `${h}:${m}`;
  };

  const handleLogout = () => {
    Alert.alert(
      t('common.logout') || "Déconnexion",
      t('common.logoutConfirm') || "Voulez-vous vraiment vous déconnecter ?",
      [
        { text: t('common.cancel'), style: "cancel" },
        { 
          text: t('common.logout'), 
          style: "destructive",
          onPress: async () => {
            await logout();
          }
        }
      ]
    );
  };

  const deviceId = telemetry?.device_id || statusPayload?.device_id || '--';
  const cameraState = telemetry?.camera?.available ? t('systemSettings.cameraAvailable') : t('common.offline');
  const lightState = telemetry?.light_sensor?.is_dark == null
    ? '--'
    : telemetry.light_sensor.is_dark
      ? t('systemSettings.lightDark')
      : t('systemSettings.lightBright');
  const lastUpdateSource = telemetry?.captured_at || statusPayload?.timestamp;
  const lastUpdate = lastUpdateSource
    ? new Date(lastUpdateSource).toLocaleTimeString()
    : '--:--:--';

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      {/* Header */}
      <BlurView intensity={30} tint="dark" style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.headerTime}>{formatTime(time)}</Text>
          <View style={styles.vDivider} />
          <Text style={styles.logoVein}>BIO</Text>
          <Text style={styles.logoGuard}>GUARD</Text>
        </View>
        <View style={styles.headerRight}>
          <TouchableOpacity style={styles.profileBtn} onPress={handleLogout}>
            <LinearGradient colors={['#1c3d5a', '#0d1b2e']} style={styles.avatarGlow}>
              <Ionicons name="person" size={16} color={COLORS.neonCyan} />
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </BlurView>

      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Welcome Section */}
        <View style={styles.welcome}>
          <View style={styles.welcomeTextBlock}>
            <Text style={styles.greeting}>{t('dashboard.title')}</Text>
            <Text style={styles.subtitle}>{t('dashboard.subtitle')}</Text>
          </View>
          <View style={[styles.systemBadge, { backgroundColor: isConnected ? 'rgba(57, 255, 20, 0.1)' : 'rgba(255, 61, 90, 0.1)', borderColor: isConnected ? 'rgba(57, 255, 20, 0.3)' : 'rgba(255, 61, 90, 0.3)' }]}>
            <Text style={[styles.systemBadgeText, { color: isConnected ? COLORS.neonGreen : COLORS.neonRed }]}>{systemStatusLabel}</Text>
          </View>
        </View>

        {/* Security Alert Banner */}
        <View style={[styles.alertBanner, { borderColor: isConnected ? 'rgba(57, 255, 20, 0.1)' : 'rgba(255, 61, 90, 0.1)' }]}>
          <Ionicons name={isConnected ? "shield-checkmark" : "warning"} size={20} color={isConnected ? COLORS.neonGreen : COLORS.neonRed} />
          <Text style={styles.alertText}>{isConnected ? t('dashboard.alertSecure') : t('dashboard.alertOffline')}</Text>
        </View>

        {/* Main Actions - Grid */}
        <View style={[styles.actionGrid, isCompact && styles.actionGridCompact]}>
          <TouchableOpacity 
            style={styles.mainAction} 
            onPress={() => navigation?.navigate('VeinScan')}
          >
            <LinearGradient colors={['rgba(188, 19, 254, 0.2)', 'rgba(138, 43, 226, 0.1)']} style={styles.actionInner}>
              <Ionicons name="hand-left" size={32} color={COLORS.neonPurple} />
              <Text style={[styles.actionLabel, { color: COLORS.neonPurple }]}>{t('dashboard.startVeinScan')}</Text>
            </LinearGradient>
          </TouchableOpacity>

          <View style={styles.sideActions}>
            <TouchableOpacity style={styles.sideAction} onPress={() => navigation?.navigate('AccessHistory')}>
              <BlurView intensity={10} style={styles.sideActionInner}>
                <Ionicons name="time" size={20} color={COLORS.neonCyan} />
                <Text numberOfLines={2} style={styles.sideActionLabel}>{t('dashboard.viewHistory')}</Text>
              </BlurView>
            </TouchableOpacity>
            <TouchableOpacity style={styles.sideAction} onPress={() => navigation?.navigate('UserManagement')}>
              <BlurView intensity={10} style={styles.sideActionInner}>
                <Ionicons name="people" size={20} color={COLORS.neonAmber} />
                <Text numberOfLines={2} style={styles.sideActionLabel}>{t('dashboard.manageUsers')}</Text>
              </BlurView>
            </TouchableOpacity>
          </View>
        </View>

        <BlurView intensity={15} tint="dark" style={styles.summaryCard}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>{t('dashboard.systemStatus')}</Text>
            <Text style={styles.nodeCount}>{systemStatusLabel}</Text>
          </View>
          <View style={[styles.summaryGrid, isCompact && styles.summaryGridCompact]}>
            <SummaryItem label={t('accessDecision.deviceLabel')} value={deviceId} accent={COLORS.neonCyan} />
            <SummaryItem label={t('systemSettings.cameraTitle')} value={cameraState} accent={telemetry?.camera?.available ? COLORS.neonGreen : COLORS.neonRed} />
            <SummaryItem label={t('systemSettings.lightSensorTitle')} value={lightState} accent={telemetry?.light_sensor?.is_dark ? COLORS.neonAmber : COLORS.neonGreen} />
            <SummaryItem label={t('dashboard.lastUpdate')} value={lastUpdate} />
          </View>
        </BlurView>

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
    paddingBottom: 15,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center' },
  headerRight: { alignItems: 'flex-end' },
  headerTime: { color: COLORS.white, fontWeight: '700', fontSize: 16 },
  vDivider: { width: 1, height: 16, backgroundColor: 'rgba(255, 255, 255, 0.2)', marginHorizontal: 12 },
  logoVein: { color: COLORS.white, fontWeight: '900', fontSize: 16, letterSpacing: 1 },
  logoGuard: { color: COLORS.neonCyan, fontWeight: '900', fontSize: 16, letterSpacing: 1 },
  profileBtn: { 
    width: 38,
    height: 38,
    borderRadius: 19,
    overflow: 'hidden',
  },
  avatarGlow: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },

  scroll: { padding: 20 },
  welcome: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 25, gap: 12 },
  welcomeTextBlock: { flex: 1, minWidth: 0 },
  greeting: { color: COLORS.white, fontSize: 28, fontWeight: '900', letterSpacing: -0.5 },
  subtitle: { color: COLORS.textSecondary, fontSize: 14, marginTop: 4, lineHeight: 20 },
  systemBadge: {
    backgroundColor: 'rgba(57, 255, 20, 0.1)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(57, 255, 20, 0.3)',
    alignSelf: 'flex-start',
    flexShrink: 1,
  },
  systemBadgeText: { color: COLORS.neonGreen, fontSize: 10, fontWeight: '800', letterSpacing: 1 },

  alertBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(57, 255, 20, 0.05)',
    padding: 15,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(57, 255, 20, 0.1)',
    marginBottom: 25,
    gap: 12,
  },
  alertText: { color: COLORS.textSecondary, fontSize: 13, fontWeight: '500', flex: 1, lineHeight: 18 },

  actionGrid: { flexDirection: 'row', gap: 15, marginBottom: 30 },
  actionGridCompact: { flexDirection: 'column' },
  mainAction: { flex: 1.2, height: 140, borderRadius: 24, overflow: 'hidden' },
  actionInner: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  actionLabel: { marginTop: 12, fontSize: 13, fontWeight: '900', textAlign: 'center', letterSpacing: 1, lineHeight: 18 },
  sideActions: { flex: 1, gap: 15 },
  sideAction: { flex: 1, borderRadius: 20, overflow: 'hidden', borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)' },
  sideActionInner: { flex: 1, justifyContent: 'center', alignItems: 'center', gap: 8, paddingHorizontal: 14, paddingVertical: 16 },
  sideActionLabel: { color: COLORS.white, fontSize: 11, fontWeight: '700', letterSpacing: 0.5, textAlign: 'center', lineHeight: 15 },

  sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 15, gap: 12 },
  sectionTitle: { color: COLORS.textDim, fontSize: 12, fontWeight: '800', letterSpacing: 2 },
  nodeCount: { color: COLORS.neonCyan, fontSize: 10, fontWeight: '700', textAlign: 'right' },

  summaryCard: {
    borderRadius: 24,
    padding: 24,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
  },
  summaryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  summaryGridCompact: {
    flexDirection: 'column',
  },
  summaryItem: {
    minHeight: 88,
    minWidth: '47%',
    flexGrow: 1,
    flexBasis: '47%',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.04)',
    gap: 8,
  },
  summaryLabel: {
    color: COLORS.textDim,
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1,
  },
  summaryValue: {
    color: COLORS.white,
    fontSize: 14,
    fontWeight: '900',
    lineHeight: 18,
  },
});
