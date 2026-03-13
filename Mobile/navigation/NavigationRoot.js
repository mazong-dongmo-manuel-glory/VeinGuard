import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { Text, View, TouchableOpacity, StyleSheet } from 'react-native';

import Login from '../ecrans/Login';
import Dashboard from '../ecrans/Dashboard';
import AccessHistory from '../ecrans/AccessHistory';
import AccessDecision from '../ecrans/AccessDecision';
import AccessEvent from '../ecrans/AccessEvent';
import SystemSetting from '../ecrans/SystemSetting';
import UserManagement from '../ecrans/UserManagement';
import VeinScanBiometrics from '../ecrans/VeinScanBiometrics';
import EnrollUser from '../ecrans/EnrollUser';

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();
const HistoryStack = createStackNavigator();
const UserStack = createStackNavigator();

const COLORS = {
  bg: '#080e1a',
  cardBg: '#0d1b2e',
  cardBorder: '#1a3a5c',
  green: '#00ff88',
  greenDark: '#002a1a',
  teal: '#00e5ff',
  amber: '#e6a020',
  textDim: '#4a6a8a',
  white: '#ffffff',
};

// ─── Icône onglet standard ─────────────────────────────────────────────────
function TabIcon({ emoji, label, focused }) {
  return (
    <View style={tabStyles.wrap}>
      <Text style={[tabStyles.emoji, focused && tabStyles.emojiFocused]}>{emoji}</Text>
    </View>
  );
}

// ─── Bouton central SCAN (FAB style) ──────────────────────────────────────
function ScanButton({ onPress }) {
  return (
    <TouchableOpacity style={tabStyles.fabWrap} onPress={onPress} activeOpacity={0.85}>
      <View style={tabStyles.fab}>
        <Text style={tabStyles.fabIcon}>〜</Text>
      </View>
    </TouchableOpacity>
  );
}

const tabStyles = StyleSheet.create({
  wrap: { alignItems: 'center', justifyContent: 'center' },
  emoji: { fontSize: 20, opacity: 0.45 },
  emojiFocused: { opacity: 1 },

  // FAB (Floating Action Button) central
  fabWrap: {
    top: -20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  fab: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: COLORS.greenDark,
    borderWidth: 2,
    borderColor: COLORS.green,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.green,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.7,
    shadowRadius: 10,
    elevation: 8,
  },
  fabIcon: { fontSize: 24, color: COLORS.green },
});

// ─── Onglets principaux (4 items + 1 FAB central) ─────────────────────────
function HistoryStackScreen() {
  return (
    <HistoryStack.Navigator screenOptions={{ headerShown: false }}>
      <HistoryStack.Screen name="AccessHistoryList" component={AccessHistory} />
      <HistoryStack.Screen name="AccessEvent" component={AccessEvent} />
    </HistoryStack.Navigator>
  );
}

function UserStackScreen() {
  return (
    <UserStack.Navigator screenOptions={{ headerShown: false }}>
      <UserStack.Screen name="UserManagementList" component={UserManagement} />
      <UserStack.Screen name="EnrollUser" component={EnrollUser} />
    </UserStack.Navigator>
  );
}

function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: COLORS.cardBg,
          borderTopColor: COLORS.cardBorder,
          borderTopWidth: 1,
          height: 68,
          paddingBottom: 8,
          paddingTop: 6,
        },
        tabBarActiveTintColor: COLORS.green,
        tabBarInactiveTintColor: COLORS.textDim,
        tabBarLabelStyle: {
          fontSize: 9,
          fontWeight: '700',
          letterSpacing: 0.5,
          marginTop: 2,
        },
      }}
    >
      {/* ① Dashboard */}
      <Tab.Screen
        name="DashboardTab"
        component={Dashboard}
        options={{
          tabBarLabel: 'ACCUEIL',
          tabBarIcon: ({ focused }) => (
            <TabIcon emoji="📊" focused={focused} />
          ),
        }}
      />

      {/* ② History */}
      <Tab.Screen
        name="AccessHistory"
        component={HistoryStackScreen}
        options={{
          tabBarLabel: 'HISTORIQUE',
          tabBarIcon: ({ focused }) => (
            <TabIcon emoji="📋" focused={focused} />
          ),
        }}
      />

      {/* ③ SCAN – bouton FAB central ─────────────────────────────────── */}
      <Tab.Screen
        name="VeinScan"
        component={VeinScanBiometrics}
        options={{
          tabBarLabel: '',
          tabBarIcon: () => null,
          tabBarButton: (props) => <ScanButton onPress={props.onPress} />,
        }}
      />

      {/* ④ Users */}
      <Tab.Screen
        name="UserManagement"
        component={UserStackScreen}
        options={{
          tabBarLabel: 'UTILISATEURS',
          tabBarIcon: ({ focused }) => (
            <TabIcon emoji="👥" focused={focused} />
          ),
        }}
      />

      {/* ⑤ Settings */}
      <Tab.Screen
        name="SystemSetting"
        component={SystemSetting}
        options={{
          tabBarLabel: 'PARAMÈTRES',
          tabBarIcon: ({ focused }) => (
            <TabIcon emoji="⚙️" focused={focused} />
          ),
        }}
      />
    </Tab.Navigator>
  );
}

// ─── Stack principal : Login → Tabs → écrans secondaires ──────────────────
function AppStack() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="Login" component={Login} />
      <Stack.Screen name="Dashboard" component={MainTabs} />
      {/* Écran de décision accessible en push depuis le scan */}
      <Stack.Screen
        name="AccessDecision"
        component={AccessDecision}
        options={{ presentation: 'modal' }}
      />
    </Stack.Navigator>
  );
}

// ─── Racine ────────────────────────────────────────────────────────────────
export default function NavigationRoot() {
  return (
    <NavigationContainer>
      <AppStack />
    </NavigationContainer>
  );
}
