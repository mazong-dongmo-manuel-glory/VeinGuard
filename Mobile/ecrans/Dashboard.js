import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
  Animated,
  Platform,
  Alert,
  useWindowDimensions,
} from 'react-native';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { COLORS, GRADIENTS, SHADOWS } from '../theme';
import { useAuthStore } from '../store/authStore';

const devices = [
  {
    id: 'BG-RPI-01',
    roleKey: 'dashboard.primaryAccessHub',
    status: 'ONLINE',
    statusColor: COLORS.neonGreen,
    heartbeatType: 'seconds',
    heartbeatValue: 2,
    rssi: '-42 dBm',
    battery: '87%',
    firmware: 'v2.4.1',
  },
  {
    id: 'BG-NODE-02',
    roleKey: 'dashboard.secondarySensorNode',
    status: 'OFFLINE',
    statusColor: COLORS.neonRed,
    heartbeatType: 'minutes',
    heartbeatValue: 5,
    rssi: '-',
    battery: '12%',
    firmware: 'v2.3.8',
  },
  {
    id: 'BG-NODE-03',
    roleKey: 'dashboard.accessControl',
    status: 'ONLINE',
    statusColor: COLORS.neonGreen,
    heartbeatType: 'seconds',
    heartbeatValue: 1,
    rssi: '-38 dBm',
    battery: 'ac_mains',
    firmware: 'v2.4.1',
  },
];

function PulsingDot({ color }) {
  const anim = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(anim, { toValue: 0.3, duration: 800, useNativeDriver: true }),
        Animated.timing(anim, { toValue: 1, duration: 800, useNativeDriver: true }),
      ])
    ).start();
  }, []);
  return (
    <Animated.View
      style={[styles.statusDot, { backgroundColor: color, opacity: anim }]}
    />
  );
}

function DeviceCard({ device }) {
  const { t } = useTranslation();
  const isOnline = device.status === 'ONLINE';
  const heartbeat =
    device.heartbeatType === 'minutes'
      ? t('dashboard.minutesAgo', { count: device.heartbeatValue })
      : t('dashboard.secondsAgo', { count: device.heartbeatValue });
  const batteryLabel = device.battery === 'ac_mains' ? t('dashboard.acMains') : device.battery;
  const displayStatus = isOnline ? t('dashboard.onlineStatus') : t('dashboard.offlineStatus');
  
  return (
    <BlurView intensity={15} tint="dark" style={[styles.deviceCard, { borderColor: isOnline ? 'rgba(0, 242, 255, 0.2)' : 'rgba(255, 61, 90, 0.2)' }]}>
      <View style={styles.deviceHeader}>
        <View style={styles.deviceHeaderMain}>
          <Text numberOfLines={1} style={[styles.deviceId, { color: isOnline ? COLORS.neonCyan : COLORS.neonRed }]}>
            {device.id}
          </Text>
          <Text numberOfLines={2} style={styles.deviceRole}>{t(device.roleKey)}</Text>
        </View>
        <PulsingDot color={isOnline ? COLORS.neonGreen : COLORS.neonRed} />
      </View>

      <View style={styles.deviceStats}>
        <View style={styles.statLine}>
          <Text style={styles.statLabel}>{t('dashboard.status')}</Text>
          <Text style={[styles.statValue, { color: device.statusColor }]}>{displayStatus}</Text>
        </View>
        <View style={styles.statGrid}>
          <View style={styles.statBox}>
            <Ionicons name="wifi" size={14} color={COLORS.textSecondary} />
            <Text style={styles.statBoxVal}>{device.rssi}</Text>
            <Text style={styles.statBoxLabel}>{t('dashboard.rssi')}</Text>
          </View>
          <View style={styles.statBox}>
            <Ionicons name="battery-dead" size={14} color={COLORS.textSecondary} />
            <Text style={styles.statBoxVal}>{batteryLabel}</Text>
            <Text style={styles.statBoxLabel}>{t('dashboard.battery')}</Text>
          </View>
          <View style={styles.statBox}>
            <Ionicons name="pulse" size={14} color={COLORS.textSecondary} />
            <Text style={styles.statBoxVal}>{heartbeat}</Text>
            <Text style={styles.statBoxLabel}>{t('dashboard.heartbeat')}</Text>
          </View>
        </View>
      </View>
    </BlurView>
  );
}

import { useMqttStore } from '../store/mqttStore';

export default function Dashboard({ navigation }) {
  const { t } = useTranslation();
  const { width } = useWindowDimensions();
  const [time, setTime] = useState(new Date());
  const isCompact = width < 390;
  
  const systemStatus = useMqttStore((state) => state.status);
  const isConnected = useMqttStore((state) => state.isConnected);
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

  const viewportWidth = Math.min(width, 600);

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

        {/* Device Section */}
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>{t('dashboard.activeNodes')}</Text>
          <Text style={styles.nodeCount}>{t('dashboard.nodeCount', { count: devices.filter((device) => device.status === 'ONLINE').length })}</Text>
        </View>

        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.deviceScroll}>
          {devices.map((d) => (
            <DeviceCard key={d.id} device={d} />
          ))}
        </ScrollView>

        {/* Broker Status */}
        <BlurView intensity={15} tint="dark" style={styles.brokerCard}>
          <View style={styles.brokerHeader}>
            <Ionicons name="radio" size={18} color={COLORS.neonCyan} />
            <Text style={styles.brokerTitle}>{t('dashboard.brokerTitle')}</Text>
            <Text style={styles.brokerStatus}>99,9 % {t('dashboard.uptime')}</Text>
          </View>
          <View style={styles.brokerStats}>
            <View style={styles.brokerStat}>
              <Text style={styles.statNum}>47</Text>
              <Text style={styles.statSubtitle}>{t('dashboard.messagesPerSecond')}</Text>
            </View>
            <View style={styles.vLine} />
            <View style={styles.brokerStat}>
              <Text style={styles.statNum}>12</Text>
              <Text style={styles.statSubtitle}>{t('dashboard.topics')}</Text>
            </View>
            <View style={styles.vLine} />
            <View style={styles.brokerStat}>
              <Text style={styles.statNum}>0</Text>
              <Text style={styles.statSubtitle}>{t('dashboard.dropped')}</Text>
            </View>
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

  deviceScroll: { paddingRight: 20, gap: 15 },
  deviceCard: {
    width: 250,
    backgroundColor: 'rgba(13, 27, 46, 0.4)',
    borderRadius: 24,
    padding: 20,
    borderWidth: 1,
  },
  deviceHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 },
  deviceHeaderMain: { flex: 1, minWidth: 0, paddingRight: 12 },
  deviceId: { fontSize: 18, fontWeight: '900', letterSpacing: 1 },
  deviceRole: { color: COLORS.textSecondary, fontSize: 10, marginTop: 4, letterSpacing: 1, lineHeight: 14 },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  
  statLine: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 15 },
  statLabel: { color: COLORS.textSecondary, fontSize: 12 },
  statValue: { fontSize: 12, fontWeight: '700' },
  statGrid: { flexDirection: 'row', justifyContent: 'space-between', backgroundColor: 'rgba(255, 255, 255, 0.03)', borderRadius: 12, padding: 12 },
  statBox: { alignItems: 'center', gap: 4, flex: 1, minWidth: 0 },
  statBoxVal: { color: COLORS.white, fontSize: 11, fontWeight: '700', textAlign: 'center' },
  statBoxLabel: { color: COLORS.textDim, fontSize: 8, fontWeight: '600' },

  brokerCard: {
    marginTop: 20,
    borderRadius: 24,
    padding: 24,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
  },
  brokerHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 20, gap: 10, flexWrap: 'wrap' },
  brokerTitle: { flex: 1, minWidth: 140, color: COLORS.white, fontSize: 12, fontWeight: '800', letterSpacing: 1, lineHeight: 16 },
  brokerStatus: { color: COLORS.neonCyan, fontSize: 10, fontWeight: '700' },
  brokerStats: { flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center' },
  brokerStat: { alignItems: 'center' },
  statNum: { color: COLORS.white, fontSize: 24, fontWeight: '900' },
  statSubtitle: { color: COLORS.textDim, fontSize: 10, fontWeight: '700', marginTop: 4 },
  vLine: { width: 1, height: 30, backgroundColor: 'rgba(255, 255, 255, 0.05)' },
});
