import Vue from 'vue'
import Router from 'vue-router'
import SemanticComm from '@/components/SemanticComm'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'SemanticComm',
      component: SemanticComm
    }
  ]
})