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

const Header = ({ navigation }) => (
  <View style={styles.header}>
    <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
      <Ionicons name="chevron-back" size={24} color={COLORS.white} />
    </TouchableOpacity>
    <Text style={styles.headerTitle}>AUTHENTICATION</Text>
    <View style={styles.spacer} />
  </View>
);

const UserHologram = ({ t }) => (
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
        <Text style={styles.userName}>JOHN MITCHELL</Text>
        <Text style={styles.userRole}>CHIEF ENGINEER · LEVEL 3</Text>
      </View>
      <View style={styles.cardDivider} />
      <View style={styles.cardGrid}>
        <View style={styles.cardCell}>
          <Text style={styles.cardLabel}>ID</Text>
          <Text style={styles.cardValue}>USR-2847</Text>
        </View>
        <View style={styles.cardCell}>
          <Text style={styles.cardLabel}>DEPT</Text>
          <Text style={styles.cardValue}>CORE OPS</Text>
        </View>
      </View>
    </BlurView>
  </View>
);

const ConfidenceMeter = ({ value }) => (
  <BlurView intensity={10} style={styles.meterCard}>
    <View style={styles.meterHeader}>
      <Text style={styles.meterLabel}>MATCH CONFIDENCE</Text>
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

export default function AccessDecision({ navigation }) {
  const { t } = useTranslation();
  const [audioOn, setAudioOn] = useState(true);

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <Header navigation={navigation} />

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        <View style={styles.statusSection}>
          <View style={styles.checkWrap}>
            <LinearGradient colors={[COLORS.neonGreen, 'transparent']} style={styles.checkGlow} />
            <View style={styles.checkInner}>
              <Ionicons name="checkmark" size={60} color={COLORS.neonGreen} />
            </View>
          </View>
          <Text style={styles.statusTitle}>ACCESS GRANTED</Text>
          <Text style={styles.statusSub}>DOOR-A12 UNLOCKED FOR 5S</Text>
        </View>

        <UserHologram t={t} />
        
        <ConfidenceMeter value={98.7} />

        <View style={styles.detailsGrid}>
          <BlurView intensity={10} style={styles.detailCard}>
            <Text style={styles.detailLabel}>DEVICE</Text>
            <Text style={styles.detailValue}>ESP32-01</Text>
          </BlurView>
          <BlurView intensity={10} style={styles.detailCard}>
            <Text style={styles.detailLabel}>TIME</Text>
            <Text style={styles.detailValue}>14:23:45</Text>
          </BlurView>
        </View>

        <View style={styles.actionsRow}>
          <TouchableOpacity style={[styles.mainBtn, styles.primaryBtn]}>
            <Text style={styles.mainBtnText}>VIEW FULL LOG</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.mainBtn, styles.secondaryBtn]}
            onPress={() => setAudioOn(!audioOn)}
          >
            <Ionicons name={audioOn ? "volume-high" : "volume-mute"} size={20} color={COLORS.white} />
            <Text style={styles.secondaryBtnText}>{audioOn ? "AUDIO ON" : "AUDIO OFF"}</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity 
          style={styles.closeBtn}
          onPress={() => navigation?.navigate('Dashboard')}
        >
          <Text style={styles.closeBtnText}>RETURN TO DASHBOARD</Text>
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
  headerTitle: { color: COLORS.white, fontSize: 16, fontWeight: '800', letterSpacing: 2 },
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
    color: COLORS.neonGreen, fontSize: 32, fontWeight: '900', letterSpacing: 2, 
    textShadowColor: COLORS.neonGreen, textShadowOffset: { width: 0, height: 0 }, textShadowRadius: 20,
    marginBottom: 8,
  },
  statusSub: { color: COLORS.textSecondary, fontSize: 12, fontWeight: '800', letterSpacing: 1 },

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
  userName: { color: COLORS.white, fontSize: 20, fontWeight: '900', letterSpacing: 1 },
  userRole: { color: COLORS.neonCyan, fontSize: 10, fontWeight: '800', letterSpacing: 1, marginTop: 5 },
  cardDivider: { height: 1, backgroundColor: 'rgba(255, 255, 255, 0.05)', marginBottom: 20 },
  cardGrid: { flexDirection: 'row', justifyContent: 'space-around' },
  cardCell: { alignItems: 'center' },
  cardLabel: { color: COLORS.textDim, fontSize: 8, fontWeight: '800', letterSpacing: 1, marginBottom: 5 },
  cardValue: { color: COLORS.white, fontSize: 13, fontWeight: '700' },

  meterCard: { borderRadius: 20, padding: 20, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)', marginBottom: 20, overflow: 'hidden' },
  meterHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 15 },
  meterLabel: { color: COLORS.textDim, fontSize: 10, fontWeight: '900', letterSpacing: 1 },
  meterValue: { fontSize: 16, fontWeight: '900' },
  barBg: { height: 8, backgroundColor: 'rgba(255, 255, 255, 0.05)', borderRadius: 4, overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 4 },

  detailsGrid: { flexDirection: 'row', gap: 15, marginBottom: 30 },
  detailCard: { flex: 1, borderRadius: 15, padding: 15, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)', overflow: 'hidden' },
  detailLabel: { color: COLORS.textDim, fontSize: 8, fontWeight: '800', letterSpacing: 1, marginBottom: 5 },
  detailValue: { color: COLORS.white, fontSize: 14, fontWeight: '700' },

  actionsRow: { flexDirection: 'row', gap: 15, marginBottom: 20 },
  mainBtn: { flex: 1, height: 55, borderRadius: 15, alignItems: 'center', justifyContent: 'center', flexDirection: 'row', gap: 10 },
  primaryBtn: { backgroundColor: COLORS.white },
  mainBtnText: { color: COLORS.bg, fontSize: 13, fontWeight: '900', letterSpacing: 1 },
  secondaryBtn: { borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.2)' },
  secondaryBtnText: { color: COLORS.white, fontSize: 12, fontWeight: '800', letterSpacing: 1 },

  closeBtn: { height: 55, borderRadius: 15, alignItems: 'center', justifyContent: 'center', borderStyle: 'dashed', borderWidth: 1, borderColor: COLORS.textDim },
  closeBtnText: { color: COLORS.textDim, fontSize: 12, fontWeight: '800', letterSpacing: 1 },
});
