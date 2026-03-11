import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { Text, View, StyleSheet } from 'react-native';

import Login from '../ecrans/Login';
import AccessHistory from '../ecrans/AccessHistory';
import AccessDecision from '../ecrans/AccessDecision';
import SystemSetting from '../ecrans/SystemSetting';
import UserManagement from '../ecrans/UserManagement';
import VeinScanBiometrics from '../ecrans/VeinScanBiometrics';

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

const COLORS = {
  bg: '#080e1a',
  cardBg: '#0d1b2e',
  cardBorder: '#1a3a5c',
  green: '#00ff88',
  teal: '#00e5ff',
  amber: '#e6a020',
  textDim: '#4a6a8a',
  white: '#ffffff',
};

// Icon components as text emojis
function TabIcon({ emoji, focused, color }) {
  return (
    <View style={[tabStyles.iconWrap, focused && tabStyles.iconWrapActive]}>
      <Text style={tabStyles.iconText}>{emoji}</Text>
    </View>
  );
}

const tabStyles = StyleSheet.create({
  iconWrap: { alignItems: 'center', justifyContent: 'center', paddingHorizontal: 4 },
  iconWrapActive: {},
  iconText: { fontSize: 18 },
});

// Main bottom tab navigator (requires authentication)
function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: COLORS.cardBg,
          borderTopColor: COLORS.cardBorder,
          borderTopWidth: 1,
          height: 60,
          paddingBottom: 6,
          paddingTop: 4,
        },
        tabBarActiveTintColor: COLORS.green,
        tabBarInactiveTintColor: COLORS.textDim,
        tabBarLabelStyle: {
          fontSize: 9,
          fontWeight: '700',
          letterSpacing: 0.5,
        },
      }}
    >
      <Tab.Screen
        name="VeinScan"
        component={VeinScanBiometrics}
        options={{
          tabBarLabel: 'SCAN',
          tabBarIcon: ({ focused, color }) => (
            <TabIcon emoji="〜" focused={focused} color={color} />
          ),
        }}
      />
      <Tab.Screen
        name="AccessHistory"
        component={AccessHistory}
        options={{
          tabBarLabel: 'HISTORY',
          tabBarIcon: ({ focused, color }) => (
            <TabIcon emoji="📋" focused={focused} color={color} />
          ),
        }}
      />
      <Tab.Screen
        name="AccessDecision"
        component={AccessDecision}
        options={{
          tabBarLabel: 'DECISION',
          tabBarIcon: ({ focused, color }) => (
            <TabIcon emoji="✅" focused={focused} color={color} />
          ),
        }}
      />
      <Tab.Screen
        name="UserManagement"
        component={UserManagement}
        options={{
          tabBarLabel: 'USERS',
          tabBarIcon: ({ focused, color }) => (
            <TabIcon emoji="👥" focused={focused} color={color} />
          ),
        }}
      />
      <Tab.Screen
        name="SystemSetting"
        component={SystemSetting}
        options={{
          tabBarLabel: 'SETTINGS',
          tabBarIcon: ({ focused, color }) => (
            <TabIcon emoji="⚙" focused={focused} color={color} />
          ),
        }}
      />
    </Tab.Navigator>
  );
}

// Root stack: Login → MainTabs
function AppStack() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="Login" component={Login} />
      <Stack.Screen name="Dashboard" component={MainTabs} />
    </Stack.Navigator>
  );
}

export default function NavigationRoot() {
  return (
    <NavigationContainer>
      <AppStack />
    </NavigationContainer>
  );
}
