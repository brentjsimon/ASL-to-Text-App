import React, { Component } from 'react';
import { AppRegistry, Platform, StatusBar, StyleSheet, View, Text, TouchableOpacity, Button, Alert } from 'react-native';
import { AppLoading, Asset, Font, Icon, Camera, Permissions } from 'expo';
import AppNavigator from './navigation/AppNavigator';

export default class ProjectApp extends Component
{
    _onPressButton()
    {
        Alert.alert('You tapped the button!')
    }
    state =
    {
        hasCameraPermission: null,
        type: null,
    }

    async cameraMount()
    {
        const { status } = await Permissions.askAsync(Permissions.CAMERA);
        this.setState( { hasCameraPermission: status === 'granted' } );
    }

    render()
    {
        const { hasCameraPermission } = this.state;
        if (hasCameraPermission === false)
        {
            return <Text> "No access to camera" </Text>;
        }
        else
        {
            return (
              <Camera style = {{ flex: 1 }} 
                      type = { this.state.type }
                      ref = { ref => { this.camera = ref; }}>
              <View style = { styles.container }>
                <View style = { styles.buttonContainer }>
                  <Button
                onPress = { () => { this.setState({ type: Camera.Constants.Type.back }); } }
                    title = "Back Camera"
                    />
                </View>
                <View style = { styles.buttonContainer }>
                    <Button
                onPress = { () => { this.setState( { type: Camera.Constants.Type.front }); } }
                    title = "Front Camera"
                    color = "#841584"
                    />
                </View>
                <View style = { styles.alternativeLayoutButtonContainer } >
                  <Button
                    onPress = { this._onPressButton } 
                    title = "Pop-up"
                    // color = "#(some number)" // to change color of text
                  />
                  <Button
                    onPress = { this._onPressButton }
                    title = "Pop-up2"
                  />
                </View>
                </View>
            </Camera>
            );
        }
    }
}

const styles = StyleSheet.create(
{
    container: 
    {
        flex: 1,
        justifyContent: 'center',
    },
    buttonContainer:
    {
        margin: 20
    },
    alternativeLayoutButtonContainer:
    {
        margin: 20,
        flexDirection: 'row',
        justifyContent: 'space-between'
    }
})

// skip this line if using Create React Native App
AppRegistry.registerComponent('AwesomeProject', () => ButtonBasics);
