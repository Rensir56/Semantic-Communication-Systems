// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import axios from 'axios'
// import VueAxios from 'vue-axios'

Vue.use(ElementUI)
Vue.prototype.$axios = axios
Vue.config.productionTip = false

// // 配置axios默认值
// axios.defaults.baseURL = 'http://localhost:3000/api'
// axios.defaults.headers.post['Content-Type'] = 'multipart/form-data'

// // 全局注册axios
// app.use(VueAxios, axios)

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
