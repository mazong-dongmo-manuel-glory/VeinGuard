import React, { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  StatusBar,
  Animated,
  Platform,
  useWindowDimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { COLORS, GRADIENTS } from '../theme';

function QualityBar({ label, value, color }) {
  const animWidth = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(animWidth, {
      toValue: value,
      duration: 1000,
      useNativeDriver: false,
    }).start();
  }, [value]);

  return (
    <View style={styles.qualityRow}>
      <View style={styles.qualityLabelRow}>
        <Text style={styles.qualityLabel}>{label.toUpperCase()}</Text>
        <Text style={[styles.qualityVal, { color }]}>{value}%</Text>
      </View>
      <View style={styles.qualityBg}>
        <Animated.View 
          style={[
            styles.qualityFill, 
            { 
              width: animWidth.interpolate({
                inputRange: [0, 100],
                outputRange: ['0%', '100%']
              }),
              backgroundColor: color 
            }
          ]} 
        />
      </View>
    </View>
  );
}

import { useMqttStore } from '../store/mqttStore';
import { Alert } from 'react-native';
import { MQTT_TOPICS } from '../config';

export default function VeinScanBiometrics({ navigation }) {
  const { t } = useTranslation();
  const { width } = useWindowDimensions();
  const [userId, setUserId] = useState('');
  const [scanning, setScanning] = useState(false);
  const scanLineAnim = useRef(new Animated.Value(0)).current;
  const isCompact = width < 390;

  const isConnected = useMqttStore((state) => state.isConnected);
  const triggerScan = useMqttStore((state) => state.triggerScan);

  useEffect(() => {
    if (scanning) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(scanLineAnim, { toValue: 1, duration: 2000, useNativeDriver: true }),
          Animated.timing(scanLineAnim, { toValue: 0, duration: 2000, useNativeDriver: true }),
        ])
      ).start();
    } else {
      scanLineAnim.setValue(0);
    }
  }, [scanning]);

  const handleScanToggle = () => {
    if (!isConnected) {
        Alert.alert(t('veinScan.offlineTitle'), t('veinScan.offlineDesc'));
        return;
    }

    if (!scanning) {
        triggerScan(userId || 'demo-user').catch(() => {
          setScanning(false);
          Alert.alert(t('veinScan.scanErrorTitle'), t('veinScan.scanErrorDesc'));
        });
        setScanning(true);
        Alert.alert(t('veinScan.scanStartedTitle'), t('veinScan.scanStartedDesc'));
    } else {
        setScanning(false);
    }
  };

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('veinScan.title')}</Text>
        <TouchableOpacity style={[styles.statusBadge, { backgroundColor: isConnected ? 'rgba(57, 255, 20, 0.1)' : 'rgba(255, 61, 90, 0.1)' }]}>
          <View style={[styles.statusDot, { backgroundColor: isConnected ? COLORS.neonGreen : COLORS.neonRed }]} />
          <Text style={[styles.statusText, { color: isConnected ? COLORS.neonGreen : COLORS.neonRed }]}>{isConnected ? t('veinScan.activeStatus') : t('common.offline')}</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.welcomeSection}>
          <Text style={styles.subtitle}>{t('veinScan.subtitle')}</Text>
        </View>

        {/* Holographic Scanner */}
        <BlurView intensity={20} tint="dark" style={styles.scannerCard}>
          <View style={styles.scannerHeader}>
            <View style={styles.recGroup}>
              <View style={[styles.recDot, !scanning && { backgroundColor: COLORS.textDim }]} />
              <Text style={[styles.recText, !scanning && { color: COLORS.textDim }]}>{scanning ? t('veinScan.liveFeed') : t('veinScan.standby')}</Text>
            </View>
            <Text style={styles.scannerId}>{t('veinScan.sensorLabel')}: BG-MULTI-01</Text>
          </View>

          <View style={styles.scannerFrame}>
            <View style={styles.cornerTL} />
            <View style={styles.cornerTR} />
            <View style={styles.cornerBL} />
            <View style={styles.cornerBR} />
            
            <View style={styles.hologramContainer}>
              <Ionicons name="hand-left" size={80} color={scanning ? COLORS.neonCyan : 'rgba(255,255,255,0.1)'} />
              {scanning && (
                <Animated.View 
                  style={[
                    styles.scanLine,
                    {
                      transform: [{
                        translateY: scanLineAnim.interpolate({
                          inputRange: [0, 1],
                          outputRange: [-60, 60]
                        })
                      }]
                    }
                  ]} 
                >
                  <LinearGradient 
                    colors={['transparent', COLORS.neonCyan, 'transparent']} 
                    start={{x: 0, y: 0.5}} 
                    end={{x: 1, y: 0.5}} 
                    style={styles.scanLineGlow} 
                  />
                </Animated.View>
              )}
            </View>

            <View style={styles.reticle}>
              <View style={styles.reticleH} />
              <View style={styles.reticleV} />
            </View>

            <View style={[styles.frameFooter, isCompact && styles.frameFooterCompact]}>
              <Text numberOfLines={1} style={styles.frameMeta}>{t('veinScan.coordinates')}: 42.0 // 18.5</Text>
              <Text numberOfLines={1} style={styles.frameMeta}>{t('veinScan.topicLabel')}: {MQTT_TOPICS.scanCmd}</Text>
            </View>
          </View>

          <View style={styles.scannerActions}>
            <TouchableOpacity 
              style={[styles.scanBtn, scanning ? styles.scanBtnActive : null]} 
              onPress={handleScanToggle}
            >
              <LinearGradient 
                colors={scanning ? [COLORS.neonRed, '#7a2020'] : GRADIENTS.neonCyan} 
                style={styles.scanBtnInner}
              >
                <Ionicons name={scanning ? "stop" : "play"} size={20} color={COLORS.white} />
                <Text style={styles.scanBtnText}>
                  {scanning ? t('veinScan.stopScan').toUpperCase() : t('veinScan.startScan').toUpperCase()}
                </Text>
              </LinearGradient>
            </TouchableOpacity>
          </View>
        </BlurView>

        {/* Intelligence & Quality */}
        <View style={styles.row}>
          <BlurView intensity={10} style={[styles.infoCard, { flex: 1 }]}>
            <Text style={styles.infoTitle}>{t('veinScan.scanQuality').toUpperCase()}</Text>
            <QualityBar label={t('veinScan.signalStrength')} value={87} color={COLORS.neonGreen} />
            <QualityBar label={t('veinScan.patternClarity')} value={64} color={COLORS.neonAmber} />
            <QualityBar label={t('veinScan.alignment')} value={92} color={COLORS.neonCyan} />
          </BlurView>
        </View>

        {/* Manual Access */}
        <BlurView intensity={10} style={styles.manualCard}>
          <Text style={styles.infoTitle}>{t('veinScan.manualIdEntry').toUpperCase()}</Text>
          <View style={[styles.inputRow, isCompact && styles.inputRowCompact]}>
            <TextInput
              style={styles.manualInput}
              placeholder={t('veinScan.userIdPlaceholder')}
              placeholderTextColor={COLORS.textDim}
              value={userId}
              onChangeText={setUserId}
            />
            <TouchableOpacity style={styles.testBtn}>
              <LinearGradient colors={['rgba(255, 216, 78, 0.2)', 'rgba(255, 216, 78, 0.05)']} style={styles.testBtnInner}>
                <Text style={styles.testBtnText}>{t('veinScan.testAccess').toUpperCase()}</Text>
              </LinearGradient>
            </TouchableOpacity>
          </View>
        </BlurView>

        {/* Security Notice */}
        <View style={styles.privacyCard}>
          <Ionicons name="shield-checkmark" size={24} color={COLORS.neonGreen} />
          <View style={styles.privacyTextContent}>
            <Text style={styles.privacyTitle}>{t('veinScan.privacyTitle').toUpperCase()}</Text>
            <Text style={styles.privacyDesc}>{t('veinScan.privacyText')}</Text>
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
  headerTitle: { color: COLORS.white, fontSize: 18, fontWeight: '800', letterSpacing: 1 },
  statusBadge: { flexDirection: 'row', alignItems: 'center', backgroundColor: 'rgba(57, 255, 20, 0.1)', paddingHorizontal: 10, paddingVertical: 5, borderRadius: 12, borderWidth: 1, borderColor: 'rgba(57, 255, 20, 0.2)', marginLeft: 8 },
  statusDot: { width: 6, height: 6, borderRadius: 3, marginRight: 6 },
  statusText: { color: COLORS.neonGreen, fontSize: 10, fontWeight: '900' },

  scroll: { flex: 1, paddingHorizontal: 20 },
  welcomeSection: { marginBottom: 20 },
  subtitle: { color: COLORS.textSecondary, fontSize: 14 },

  scannerCard: {
    borderRadius: 30,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    marginBottom: 20,
    overflow: 'hidden',
  },
  scannerHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, gap: 12 },
  recGroup: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  recDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: COLORS.neonRed },
  recText: { color: COLORS.neonRed, fontSize: 10, fontWeight: '900', letterSpacing: 1 },
  scannerId: { color: COLORS.textDim, fontSize: 9, fontWeight: '700', flexShrink: 1, textAlign: 'right' },

  scannerFrame: {
    height: 240,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: 24,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
    overflow: 'hidden',
  },
  cornerTL: { position: 'absolute', top: 15, left: 15, width: 25, height: 25, borderTopWidth: 2, borderLeftWidth: 2, borderColor: COLORS.neonCyan },
  cornerTR: { position: 'absolute', top: 15, right: 15, width: 25, height: 25, borderTopWidth: 2, borderRightWidth: 2, borderColor: COLORS.neonCyan },
  cornerBL: { position: 'absolute', bottom: 15, left: 15, width: 25, height: 25, borderBottomWidth: 2, borderLeftWidth: 2, borderColor: COLORS.neonCyan },
  cornerBR: { position: 'absolute', bottom: 15, right: 15, width: 25, height: 25, borderBottomWidth: 2, borderRightWidth: 2, borderColor: COLORS.neonCyan },
  
  hologramContainer: { width: 120, height: 120, borderRadius: 60, backgroundColor: 'rgba(0, 242, 255, 0.05)', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: 'rgba(0, 242, 255, 0.1)' },
  scanLine: { position: 'absolute', width: '100%', height: 2 },
  scanLineGlow: { flex: 1 },

  reticle: { position: 'absolute', width: 40, height: 40, justifyContent: 'center', alignItems: 'center' },
  reticleH: { width: '100%', height: 1, backgroundColor: 'rgba(255, 255, 255, 0.1)' },
  reticleV: { height: '100%', width: 1, backgroundColor: 'rgba(255, 255, 255, 0.1)' },

  frameFooter: { position: 'absolute', bottom: 15, left: 0, right: 0, flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 20, gap: 12 },
  frameFooterCompact: { flexDirection: 'column', alignItems: 'flex-start' },
  frameMeta: { color: 'rgba(255, 255, 255, 0.2)', fontSize: 8, fontWeight: '700', letterSpacing: 1, flexShrink: 1 },

  scannerActions: { marginTop: 20 },
  scanBtn: { borderRadius: 18, overflow: 'hidden' },
  scanBtnInner: { paddingVertical: 18, flexDirection: 'row', justifyContent: 'center', alignItems: 'center', gap: 10 },
  scanBtnText: { color: COLORS.white, fontWeight: '900', letterSpacing: 2, fontSize: 14 },

  infoCard: { borderRadius: 24, padding: 20, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)', marginBottom: 20 },
  infoTitle: { color: COLORS.textDim, fontSize: 10, fontWeight: '900', letterSpacing: 2, marginBottom: 20 },

  qualityRow: { marginBottom: 15 },
  qualityLabelRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
  qualityLabel: { color: COLORS.textSecondary, fontSize: 10, fontWeight: '700' },
  qualityVal: { fontSize: 11, fontWeight: '800' },
  qualityBg: { height: 6, backgroundColor: 'rgba(255, 255, 255, 0.03)', borderRadius: 3, overflow: 'hidden' },
  qualityFill: { height: '100%', borderRadius: 3 },

  manualCard: { borderRadius: 24, padding: 20, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)', marginBottom: 20 },
  inputRow: { flexDirection: 'row', gap: 10 },
  inputRowCompact: { flexDirection: 'column' },
  manualInput: { flex: 1, backgroundColor: 'rgba(255, 255, 255, 0.03)', borderRadius: 12, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.08)', color: COLORS.white, paddingHorizontal: 15, paddingVertical: 12, fontSize: 14 },
  testBtn: { borderRadius: 12, overflow: 'hidden', borderWidth: 1, borderColor: 'rgba(255, 216, 78, 0.2)' },
  testBtnInner: { paddingHorizontal: 15, justifyContent: 'center', flex: 1 },
  testBtnText: { color: COLORS.neonAmber, fontSize: 10, fontWeight: '900', letterSpacing: 1 },

  privacyCard: { flexDirection: 'row', backgroundColor: 'rgba(57, 255, 20, 0.03)', borderRadius: 20, padding: 15, gap: 15, borderWidth: 1, borderColor: 'rgba(57, 255, 20, 0.1)', alignItems: 'flex-start' },
  privacyTextContent: { flex: 1 },
  privacyTitle: { color: COLORS.neonGreen, fontSize: 11, fontWeight: '900', letterSpacing: 1, marginBottom: 4 },
  privacyDesc: { color: COLORS.textDim, fontSize: 12, lineHeight: 18 },
});
