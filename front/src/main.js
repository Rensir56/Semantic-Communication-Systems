import Vue from 'vue'
import App from './App'
import router from './router'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import axios from 'axios'

Vue.use(ElementUI)

// 配置 axios
axios.defaults.baseURL = 'http://localhost:5000'  // 设置基础URL
axios.defaults.headers.post['Content-Type'] = 'multipart/form-data'
Vue.prototype.$axios = axios

Vue.config.productionTip = false

new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})