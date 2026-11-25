import { mount } from 'svelte'
import App from './App.svelte'
import { Session } from './lib/session'
import './style.css'

const target = document.getElementById('app')
if (target) mount(App, { target, props: { session: new Session(), pathname: location.pathname } })
