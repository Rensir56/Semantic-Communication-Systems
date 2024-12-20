<template>
  <div class="semantic-comm">
    <h2 class="title">语义通信系统</h2>
    <el-row :gutter="20">
      <el-col :span="12">
        <div class="left-panel">
          <div class="image-box">
            <h3>原始图片</h3>
            <!-- 上传区域 -->
            <div class="upload-area" v-if="!originalImages.length">
              <el-upload
                class="uploader"
                action="#"
                :auto-upload="false"
                :show-file-list="false"
                :on-change="handleImageSelect"
                multiple
                accept="image/*"
              >
                <el-button type="primary" size="large">
                  <i class="el-icon-plus"></i> 选择图片
                </el-button>
                <div class="upload-tip">请选择图片</div>
              </el-upload>
            </div>
            <!-- 图片预览区域 -->
            <div class="image-preview" v-else>
              <div class="image-grid">
                <div v-for="(img, index) in originalImages" 
                      :key="index" 
                      class="preview-item">
                  <img :src="img" :alt="'Original ' + (index + 1)">
                  <div class="image-actions">
                    <i class="el-icon-delete" @click="removeImage(index)"></i>
                  </div>
                </div>
              </div>
              <el-button 
                v-if="originalImages.length" 
                type="warning" 
                class="reset-button"
                @click="resetImages"
              >
                重新选择图片
              </el-button>
            </div>
          </div>

          <!-- UAP开关 -->
          <div class="uap-switches">
            <el-radio-group v-model="uapMode" @change="handleUapModeChange">
              <el-radio label="none">关闭UAP</el-radio>
              <el-radio label="white">白盒UAP攻击</el-radio>
              <el-radio label="black">黑盒UAP攻击</el-radio>
            </el-radio-group>
          </div>
          
          <!-- 发送按钮 -->
          <div class="button-group">
            <el-button 
              type="primary" 
              @click="sendImages"
              :disabled="!originalImages.length || originalImages.length !== 16"
              :loading="sendingImages"
            >
              发送图片
            </el-button>
            <el-button 
              type="success" 
              @click="handleReconstructAndEvaluate"
              :disabled="!hasCompressed"
              :loading="processing"
            >
              重构和评估
            </el-button>
          </div>
        </div>
      </el-col>

      <el-col :span="12">  
        <div class="right-panel">
          <!-- 普通重构结果 -->
          <div class="image-box" v-if="hasReceivedImages">
            <h3>重构识别结果</h3>
            <div class="image-preview">
              <div class="image-grid">
                <div v-for="(img, index) in reconstructedImages" 
                      :key="index" 
                      class="preview-item">
                  <img :src="img" :alt="'Reconstructed ' + (index + 1)">
                </div>
              </div>
            </div>
            <div class="evaluation-results">
              <p>重构识别准确率: {{ accuracy/100 }}%</p>
              <p>预测值: {{ predictions.join(', ') }}</p>
              <p>真实值: {{ trueLabels.join(', ') }}</p>
            </div>
          </div>
        </div>
      </el-col>
    </el-row>

    <el-row :gutter="20" v-if="uapMode !== 'none'">
      <el-col :span="12">
        <!-- 白盒UAP结果 -->
        <div class="image-box">
          <h3>白盒UAP攻击结果</h3>
          <div v-if="!hasWhiteUapImages" class="waiting-text">
            等待上传图片...
          </div>
          <template v-else>
            <div class="image-preview">
              <div class="image-grid">
                <div v-for="(img, index) in whiteUapImages" 
                     :key="index" 
                     class="preview-item">
                  <img :src="img" :alt="'White UAP ' + (index + 1)">
                </div>
              </div>
            </div>
            <div class="evaluation-results">
              <p>重构识别准确率: {{ whiteUapAccuracy/100 }}%</p>
              <p>预测值: {{ whiteUapPredictions.join(', ') }}</p>
              <p>真实值: {{ whiteUapTrueLabels.join(', ') }}</p>
            </div>
          </template>
        </div>
      </el-col>
      
      <el-col :span="12">
        <!-- 黑盒UAP结果 -->
        <div class="image-box">
          <h3>黑盒UAP攻击结果</h3>
          <div v-if="!hasBlackUapImages" class="waiting-text">
            等待上传图片...
          </div>
          <template v-else>
            <div class="image-preview">
              <div class="image-grid">
                <div v-for="(img, index) in blackUapImages" 
                     :key="index" 
                     class="preview-item">
                  <img :src="img" :alt="'Black UAP ' + (index + 1)">
                </div>
              </div>
            </div>
            <div class="evaluation-results">
              <p>重构识别准确率: {{ blackUapAccuracy/100 }}%</p>
              <p>预测值: {{ blackUapPredictions.join(', ') }}</p>
              <p>真实值: {{ blackUapTrueLabels.join(', ') }}</p>
            </div>
          </template>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script>
export default {
  name: 'SemanticComm',
  data() {
    return {
      originalImages: [],
      reconstructedImages: [],      // 普通重构图片
      uapImages: [],               // UAP重构图片
      hasCompressed: false,
      hasReceivedImages: false,
      hasUapImages: false,         // UAP图片接收状态
      sendingImages: false,
      processing: false,
      accuracy: null,
      predictions: [],
      trueLabels: [],
      uap_accuracy: null,
      uap_predictions: [],
      uap_trueLabels: [],
      uapEnabled: false,
      uapMode: 'none', // 'none' | 'white' | 'black'
      hasWhiteUapImages: false,
      hasBlackUapImages: false,
      whiteUapImages: [],
      blackUapImages: [],
      whiteUapAccuracy: null,
      whiteUapPredictions: [],
      whiteUapTrueLabels: [],
      blackUapAccuracy: null,
      blackUapPredictions: [],
      blackUapTrueLabels: [],
    }
  },
  methods: {
    handleImageSelect(file) {
      if (this.originalImages.length >= 16) {
        this.$message.warning('最多只能选择16张图片')
        return
      }

      const reader = new FileReader()
      reader.onload = (e) => {
        this.originalImages.push(e.target.result)
      }
      reader.readAsDataURL(file.raw)
    },

    removeImage(index) {
      this.originalImages.splice(index, 1)
    },

    async sendImages() {
      if (this.originalImages.length !== 16) {
        this.$message.warning('请确保选择了16张图片')
        return
      }
        this.sendingImages = true
      try {
        const response = await this.$axios.post('/api/compress')
        if (response.data.status === 'success') {
          this.hasCompressed = true  // 压缩成功后设置标记
          this.$message.success('图片压缩成功')
        }
      } catch (error) {
        console.error('Error:', error)
        this.$message.error('压缩失败: ' + error.message)
        this.hasCompressed = false
      } finally {
        this.sendingImages = false
      }
    },

    async handleReconstructAndEvaluate() {
      if (!this.hasCompressed && this.uapMode !== 'black') {
        this.$message.warning('请先发送并压缩图片')
        return
      }
      
      this.processing = true
      try {
        const response = await this.$axios.post('/api/reconstruct-and-evaluate', null, {
          params: {
            uapMode: this.uapMode
          }
        })

        if (response.data.status === 'success') {
          if (this.uapMode === 'white') {
            this.whiteUapImages = response.data.reconstructed_images
            this.hasWhiteUapImages = true
            this.whiteUapAccuracy = response.data.accuracy
            this.whiteUapPredictions = response.data.predictions
            this.whiteUapTrueLabels = response.data.true_labels
          } else if (this.uapMode === 'black') {
            this.blackUapImages = response.data.reconstructed_images
            this.hasBlackUapImages = true
            this.blackUapAccuracy = response.data.accuracy
            this.blackUapPredictions = response.data.predictions
            this.blackUapTrueLabels = response.data.true_labels
          } else {
            this.reconstructedImages = response.data.reconstructed_images
            this.hasReceivedImages = true
            this.accuracy = response.data.accuracy
            this.predictions = response.data.predictions
            this.trueLabels = response.data.true_labels
          }
          this.$message.success('重构和评估完成')
        }
      } catch (error) {
        console.error('Error:', error)
        this.$message.error('重构和评估失败: ' + error.message)
      } finally {
        this.processing = false
      }
    },

    handleUapModeChange(mode) {
      // 不需要重置状态
    },

    resetImages() {
      this.originalImages = []
      this.hasCompressed = false
      // 不重置UAP相关的状态
    },

    base64ToBlob(base64) {
      const parts = base64.split(';base64,')
      const contentType = parts[0].split(':')[1]
      const raw = window.atob(parts[1])
      const rawLength = raw.length
      const uInt8Array = new Uint8Array(rawLength)

      for (let i = 0; i < rawLength; ++i) {
        uInt8Array[i] = raw.charCodeAt(i)
      }

      return new Blob([uInt8Array], { type: contentType })
    }
  }
}
</script>

<style scoped>
.semantic-comm {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.title {
  text-align: center;
  margin-bottom: 30px;
  color: #303133;
  font-size: 42px;
  font-weight: bold;
}

.left-panel, .right-panel {
  height: 100%;
}

.upload-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  border: 2px dashed #dcdfe6;
  border-radius: 4px;
  background-color: #fafafa;
  transition: all 0.3s;
}

.upload-area:hover {
  border-color: #409eff;
  background-color: #ecf5ff;
}

.upload-tip {
  margin-top: 10px;
  color: #909399;
  font-size: 16px;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr); /* 4列 */
  grid-template-rows: repeat(4, 1fr);    /* 4行 */
  gap: 10px;
  width: 100%;  /* 使用容器的全宽 */
  aspect-ratio: 1;  /* 保持正方形 */
}

.image-box {
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  padding: 20px;
  margin-bottom: 20px;
  background-color: #fff;
}

.image-box h3 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #303133;
  font-size: 28px;
  font-weight: bold;
}

.image-preview {
  position: relative;
  width: 100%;
  aspect-ratio: 1;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  overflow: hidden;
  background-color: #fff;
}

.preview-item {
  position: relative;
  aspect-ratio: 1;
  overflow: hidden;
  border: 1px solid #ebeef5;
  border-radius: 4px;
}

.preview-item img {
  width: 100%;
  height: 100%;
  padding: 5px;
  object-fit: contain;
}

.image-actions {
  position: absolute;
  top: 5px;
  right: 5px;
  background-color: rgba(0, 0, 0, 0.6);
  border-radius: 50%;
  width: 24px;
  height: 24px;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.3s;
}

.preview-item:hover .image-actions {
  opacity: 1;
}

.image-actions i {
  color: #fff;
  font-size: 16px;
}

.uap-switches {
  margin: 20px 0;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.button-group {
  margin: 20px 0;
  text-align: center;
}

.button-group .el-button {
  padding: 12px 25px;
  font-size: 18px;
  font-weight: bold;
}

.evaluation-results {
  margin-top: 20px;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.uap-results {
  margin-top: 20px;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.accuracy-info {
  margin-top: 10px;
  padding: 10px;
  background-color: #f5f7fa;
  border-radius: 4px;
  text-align: center;
}

.accuracy-info p {
  margin: 0;
  color: #409eff;
  font-weight: bold;
  font-size: 20px;
}

.reset-button {
  position: sticky;
  bottom: 10px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 1;
  margin-top: 10px;
}

/* Element UI 组件样式覆盖 */
:deep(.el-switch__label) {
  font-size: 24px !important;
  font-weight: 500;
}

:deep(.el-switch__label.is-active) {
  color: #409eff !important;
  font-weight: bold;
}

:deep(.el-switch) {
  height: 32px;
}

  :deep(.el-switch__core) {
    height: 32px !important;  /* 增大开关核心高度 */
    width: 60px !important;  /* 增大开关宽度 */
  }
  </style>