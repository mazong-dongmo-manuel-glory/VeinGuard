import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
  Dimensions,
  Platform,
} from 'react-native';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { COLORS, GRADIENTS } from '../theme';

const { width } = Dimensions.get('window');

const Header = ({ navigation, t }) => (
  <View style={styles.header}>
    <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
      <Ionicons name="chevron-back" size={24} color={COLORS.white} />
    </TouchableOpacity>
    <Text style={styles.headerTitle}>{t('accessDecision.headerTitle')}</Text>
    <View style={styles.spacer} />
  </View>
);

const formatEventScore = (value) => {
  const numericValue =
    typeof value === 'number'
      ? value
      : typeof value === 'string'
        ? Number.parseFloat(value.replace('%', ''))
        : Number.NaN;

  if (!Number.isFinite(numericValue)) {
    return 98.7;
  }

  return numericValue <= 1 ? numericValue * 100 : numericValue;
};

const formatEventTime = (timestamp) => {
  if (!timestamp) {
    return '--:--:--';
  }

  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return String(timestamp);
  }

  return date.toLocaleTimeString();
};

const UserHologram = ({ t, userName, userMeta, userId, department }) => (
  <View style={styles.hologramContainer}>
    <LinearGradient colors={['rgba(57, 255, 20, 0.15)', 'transparent']} style={styles.hologramBeam} />
    <BlurView intensity={20} tint="dark" style={styles.userCard}>
      <View style={styles.avatarWrap}>
        <LinearGradient colors={[COLORS.neonGreen, COLORS.neonCyan]} style={styles.avatarGlow} />
        <View style={styles.avatarInner}>
          <Ionicons name="person" size={40} color={COLORS.white} />
        </View>
      </View>
      <View style={styles.userInfo}>
        <Text numberOfLines={1} style={styles.userName}>{userName}</Text>
        <Text numberOfLines={2} style={styles.userRole}>{userMeta}</Text>
      </View>
      <View style={styles.cardDivider} />
      <View style={styles.cardGrid}>
        <View style={styles.cardCell}>
          <Text style={styles.cardLabel}>{t('userManagement.idLabel')}</Text>
          <Text style={styles.cardValue}>{userId}</Text>
        </View>
        <View style={styles.cardCell}>
          <Text style={styles.cardLabel}>{t('accessDecision.departmentShort')}</Text>
          <Text style={styles.cardValue}>{department}</Text>
        </View>
      </View>
    </BlurView>
  </View>
);

const ConfidenceMeter = ({ value, t }) => (
  <BlurView intensity={10} style={styles.meterCard}>
    <View style={styles.meterHeader}>
      <Text style={styles.meterLabel}>{t('accessDecision.matchConfidence')}</Text>
      <Text style={[styles.meterValue, { color: COLORS.neonGreen }]}>{value}%</Text>
    </View>
    <View style={styles.barBg}>
      <LinearGradient
        colors={[COLORS.neonCyan, COLORS.neonGreen]}
        start={[0, 0]}
        end={[1, 0]}
        style={[styles.barFill, { width: `${value}%` }]}
      />
    </View>
  </BlurView>
);

export default function AccessDecision({ navigation, route }) {
  const { t } = useTranslation();
  const [audioOn, setAudioOn] = useState(true);
  const isCompact = width < 390;
  const event = route?.params?.event || {};
  const userName = event?.username || t('common.unknownUser');
  const userMeta = String(event?.method || event?.reason || t('accessDecision.userRoleValue')).toUpperCase();
  const userId = event?.user_id || '--';
  const department = event?.department || '--';
  const deviceId = event?.device_id || 'BG-RPI-01';
  const eventTime = formatEventTime(event?.timestamp);
  const eventScore = formatEventScore(event?.score);
  const isGranted = String(event?.status || 'GRANTED').toUpperCase() === 'GRANTED';
  const statusTitle = isGranted ? t('accessDecision.statusGranted') : t('accessHistory.denied');

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <Header navigation={navigation} t={t} />

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        <View style={styles.statusSection}>
          <View style={styles.checkWrap}>
            <LinearGradient colors={[COLORS.neonGreen, 'transparent']} style={styles.checkGlow} />
            <View style={styles.checkInner}>
              <Ionicons name={isGranted ? "checkmark" : "close"} size={60} color={isGranted ? COLORS.neonGreen : COLORS.neonRed} />
            </View>
          </View>
          <Text style={[styles.statusTitle, !isGranted && styles.statusTitleDenied]}>{statusTitle}</Text>
          <Text style={styles.statusSub}>{t('accessDecision.doorUnlockedFor5s')}</Text>
        </View>

        <UserHologram
          t={t}
          userName={userName}
          userMeta={userMeta}
          userId={userId}
          department={department}
        />
        
        <ConfidenceMeter value={eventScore} t={t} />

        <View style={[styles.detailsGrid, isCompact && styles.detailsGridCompact]}>
          <BlurView intensity={10} style={styles.detailCard}>
            <Text style={styles.detailLabel}>{t('accessDecision.deviceLabel')}</Text>
            <Text style={styles.detailValue}>{deviceId}</Text>
          </BlurView>
          <BlurView intensity={10} style={styles.detailCard}>
            <Text style={styles.detailLabel}>{t('accessDecision.timeLabel')}</Text>
            <Text style={styles.detailValue}>{eventTime}</Text>
          </BlurView>
        </View>

        <View style={[styles.actionsRow, isCompact && styles.actionsRowCompact]}>
          <TouchableOpacity style={[styles.mainBtn, styles.primaryBtn]}>
            <Text style={styles.mainBtnText}>{t('accessDecision.viewFullLog')}</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.mainBtn, styles.secondaryBtn]}
            onPress={() => setAudioOn(!audioOn)}
          >
            <Ionicons name={audioOn ? "volume-high" : "volume-mute"} size={20} color={COLORS.white} />
            <Text style={styles.secondaryBtnText}>{audioOn ? t('accessDecision.audioEnabled') : t('accessDecision.audioDisabled')}</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity 
          style={styles.closeBtn}
          onPress={() => navigation?.navigate('Dashboard')}
        >
          <Text style={styles.closeBtnText}>{t('accessDecision.returnDashboard')}</Text>
        </TouchableOpacity>

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
  headerTitle: { color: COLORS.white, fontSize: 15, fontWeight: '800', letterSpacing: 1.5 },
  spacer: { width: 40 },

  scroll: { flex: 1, paddingHorizontal: 25 },
  scrollContent: { paddingTop: 10 },

  statusSection: { alignItems: 'center', marginBottom: 40 },
  checkWrap: { width: 120, height: 120, alignItems: 'center', justifyContent: 'center', marginBottom: 20 },
  checkGlow: { ...StyleSheet.absoluteFillObject, borderRadius: 60, opacity: 0.3 },
  checkInner: { 
    width: 100, height: 100, borderRadius: 50, 
    borderWidth: 2, borderColor: COLORS.neonGreen, 
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: 'rgba(57, 255, 20, 0.05)',
  },
  statusTitle: { 
    color: COLORS.neonGreen, fontSize: 28, fontWeight: '900', letterSpacing: 1.5,
    textShadowColor: COLORS.neonGreen, textShadowOffset: { width: 0, height: 0 }, textShadowRadius: 20,
    marginBottom: 8,
    textAlign: 'center',
  },
  statusSub: { color: COLORS.textSecondary, fontSize: 12, fontWeight: '800', letterSpacing: 1, textAlign: 'center', lineHeight: 18 },
  statusTitleDenied: {
    color: COLORS.neonRed,
    textShadowColor: COLORS.neonRed,
  },

  hologramContainer: { marginBottom: 30, alignItems: 'center' },
  hologramBeam: { position: 'absolute', top: -40, width: 2, height: 200, opacity: 0.5 },
  userCard: { 
    width: '100%', borderRadius: 30, padding: 25, 
    borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.1)',
    overflow: 'hidden',
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
  },
  avatarWrap: { width: 80, height: 80, alignSelf: 'center', marginBottom: 15 },
  avatarGlow: { ...StyleSheet.absoluteFillObject, borderRadius: 40, opacity: 0.2 },
  avatarInner: { 
    width: 80, height: 80, borderRadius: 40, 
    borderWidth: 1, borderColor: COLORS.neonCyan, 
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: 'rgba(0, 243, 255, 0.05)',
  },
  userInfo: { alignItems: 'center', marginBottom: 20 },
  userName: { color: COLORS.white, fontSize: 20, fontWeight: '900', letterSpacing: 1, textAlign: 'center' },
  userRole: { color: COLORS.neonCyan, fontSize: 10, fontWeight: '800', letterSpacing: 1, marginTop: 5, textAlign: 'center', lineHeight: 15 },
  cardDivider: { height: 1, backgroundColor: 'rgba(255, 255, 255, 0.05)', marginBottom: 20 },
  cardGrid: { flexDirection: 'row', justifyContent: 'space-around', gap: 12 },
  cardCell: { alignItems: 'center', flex: 1, minWidth: 0 },
  cardLabel: { color: COLORS.textDim, fontSize: 8, fontWeight: '800', letterSpacing: 1, marginBottom: 5 },
  cardValue: { color: COLORS.white, fontSize: 13, fontWeight: '700', textAlign: 'center' },

  meterCard: { borderRadius: 20, padding: 20, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)', marginBottom: 20, overflow: 'hidden' },
  meterHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 15 },
  meterLabel: { color: COLORS.textDim, fontSize: 10, fontWeight: '900', letterSpacing: 1 },
  meterValue: { fontSize: 16, fontWeight: '900' },
  barBg: { height: 8, backgroundColor: 'rgba(255, 255, 255, 0.05)', borderRadius: 4, overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 4 },

  detailsGrid: { flexDirection: 'row', gap: 15, marginBottom: 30 },
  detailsGridCompact: { flexDirection: 'column' },
  detailCard: { flex: 1, borderRadius: 15, padding: 15, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)', overflow: 'hidden' },
  detailLabel: { color: COLORS.textDim, fontSize: 8, fontWeight: '800', letterSpacing: 1, marginBottom: 5 },
  detailValue: { color: COLORS.white, fontSize: 14, fontWeight: '700' },

  actionsRow: { flexDirection: 'row', gap: 15, marginBottom: 20 },
  actionsRowCompact: { flexDirection: 'column' },
  mainBtn: { flex: 1, minHeight: 55, borderRadius: 15, alignItems: 'center', justifyContent: 'center', flexDirection: 'row', gap: 10, paddingHorizontal: 16, paddingVertical: 14 },
  primaryBtn: { backgroundColor: COLORS.white },
  mainBtnText: { color: COLORS.bg, fontSize: 13, fontWeight: '900', letterSpacing: 1, textAlign: 'center' },
  secondaryBtn: { borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.2)' },
  secondaryBtnText: { color: COLORS.white, fontSize: 12, fontWeight: '800', letterSpacing: 1, textAlign: 'center', flexShrink: 1 },

  closeBtn: { height: 55, borderRadius: 15, alignItems: 'center', justifyContent: 'center', borderStyle: 'dashed', borderWidth: 1, borderColor: COLORS.textDim },
  closeBtnText: { color: COLORS.textDim, fontSize: 12, fontWeight: '800', letterSpacing: 1, textAlign: 'center' },
});
