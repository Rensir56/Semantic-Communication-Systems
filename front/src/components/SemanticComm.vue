<template>
  <div class="semantic-comm">
    <h2 class="title">语义通信系统</h2>
    <el-row :gutter="20">
      <!-- 左侧：原始图片上传区域和UAP开关 -->
      <el-col :span="12">
        <div class="left-panel">
          <!-- 原始图片上传区域 -->
          <div class="image-box">
            <h3>原始图片</h3>
            <div class="image-container" @click="triggerUpload">
              <template v-if="!selectedImages.length">
                <i class="el-icon-plus"></i>
                <p>点击选择图片</p>
              </template>
              <div class="image-preview" v-else>
                <div
                  v-for="(img, index) in selectedImages"
                  :key="index"
                  class="preview-item"
                >
                  <img :src="img.url" :alt="'Image ' + (index + 1)" />
                  <div class="image-actions">
                    <i
                      class="el-icon-delete"
                      @click.stop="removeImage(index)"
                    ></i>
                  </div>
                </div>
                <div
                  class="add-more"
                  @click.stop="triggerUpload"
                  v-if="selectedImages.length < 16"
                >
                  <i class="el-icon-plus"></i>
                  <p>添加更多</p>
                </div>
              </div>
            </div>
            <input
              type="file"
              ref="fileInput"
              multiple
              accept="image/*"
              style="display: none"
              @change="handleFileSelect"
            />
          </div>

          <!-- UAP开关 -->
          <div class="uap-switch">
            <el-switch
              v-model="uapEnabled"
              active-text="启用UAP对抗样本扰动"
              inactive-text="关闭UAP"
              @change="handleUapChange"
            ></el-switch>
          </div>

          <!-- 按钮组 -->
          <div class="button-group">
            <el-button
              type="primary"
              @click="sendImages"
              :disabled="!selectedImages.length"
            >
              发送图片
            </el-button>
            <el-button
              type="success"
              @click="evaluate"
              :disabled="!hasReceivedImages"
            >
              评估结果
            </el-button>
          </div>
        </div>
      </el-col>

      <!-- 右侧：两个图片显示区域 -->
      <el-col :span="12">
        <div class="right-panel">
          <!-- 重构图片区域 -->
          <div class="image-box">
            <h3>重构图片</h3>
            <div class="image-container" v-if="!hasReceivedImages">
              <p>等待接收图片...</p>
            </div>
            <div class="image-preview" v-else>
              <div
                v-for="(img, index) in reconstructedImages"
                :key="index"
                class="preview-item"
              >
                <img :src="img" :alt="'Reconstructed ' + (index + 1)" />
              </div>
            </div>
            <div class="accuracy-info" v-if="reconstructedAccuracy !== null">
              <p>重构识别准确率: {{ reconstructedAccuracy }}%</p>
            </div>
          </div>

          <!-- UAP处理后的图片区域 -->
          <div class="image-box" v-if="uapEnabled">
            <h3>UAP对抗样本处理后的图片</h3>
            <div class="image-container" v-if="!hasUapImages">
              <p>等待UAP处理...</p>
            </div>
            <div class="image-preview" v-else>
              <div
                v-for="(img, index) in uapImages"
                :key="index"
                class="preview-item"
              >
                <img :src="img" :alt="'UAP ' + (index + 1)" />
              </div>
            </div>
            <div class="accuracy-info" v-if="uapAccuracy !== null">
              <p>UAP对抗后识别准确率: {{ uapAccuracy }}%</p>
            </div>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script>
export default {
  name: "SemanticComm",
  data() {
    return {
      selectedImages: [],
      reconstructedImages: [],
      uapImages: [],
      hasReceivedImages: false,
      hasUapImages: false,
      uapEnabled: false,
      reconstructedAccuracy: null,
      uapAccuracy: null,
    };
  },
  methods: {
    triggerUpload() {
      this.$refs.fileInput.click();
    },
    handleFileSelect(event) {
      const files = Array.from(event.target.files);
      const totalImages = this.selectedImages.length + files.length;

      if (totalImages > 16) {
        this.$message.error("最多只能选择16张图片");
        return;
      }

      const newImages = files.map((file) => ({
        file,
        url: URL.createObjectURL(file),
      }));

      this.selectedImages.push(...newImages);
      event.target.value = "";
    },
    removeImage(index) {
      // 释放已创建的 URL 对象
      URL.revokeObjectURL(this.selectedImages[index].url);
      this.selectedImages.splice(index, 1);
    },
    async sendImages() {
      const formData = new FormData();
      this.selectedImages.forEach((img, index) => {
        formData.append("images", img.file);
      });

      try {
        const response = await this.$axios.post("/compress", formData, {
          params: {
            uap_enabled: this.uapEnabled,
          },
        });
        this.hasReceivedImages = true;
        this.reconstructedImages = response.data.reconstructed_images;
        if (this.uapEnabled && response.data.uap_images) {
          this.hasUapImages = true;
          this.uapImages = response.data.uap_images;
        }
        this.$message.success("图片发送成功");
      } catch (error) {
        console.error("Error:", error);
        this.$message.error("图片处理失败: " + error.message);
      }
    },
    async evaluate() {
      try {
        const response = await this.$axios.get("/evaluate", {
          params: {
            uap_enabled: this.uapEnabled,
          },
        });
        this.reconstructedAccuracy = response.data.reconstructed_accuracy;
        if (this.uapEnabled) {
          this.uapAccuracy = response.data.uap_accuracy;
        }
        this.$message.success("评估完成");
      } catch (error) {
        console.error("Error:", error);
        this.$message.error("评估失败: " + error.message);
      }
    },
    async handleUapChange(value) {
      this.uapEnabled = value;
      // 如果关闭UAP,清除相关数据
      if (!value) {
        this.hasUapImages = false;
        this.uapImages = [];
        this.uapAccuracy = null;
      }
    },
  },
};
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
  font-size: 42px; /* 增大标题字体 */
  font-weight: bold;
}

.left-panel,
.right-panel {
  height: 100%;
}

.uap-switch {
  margin: 20px 0;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
  font-size: 24px; /* 增大开关文字 */
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
  font-size: 28px; /* 增大子标题字体 */
  font-weight: bold;
}

.image-container {
  min-height: 300px;
  border: 2px dashed #dcdfe6;
  border-radius: 4px;
  padding: 10px;
  background-color: #fafafa;
}

.image-container:hover {
  border-color: #409eff;
}

.image-container:not(.has-images) {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  cursor: pointer;
}

.image-container i {
  font-size: 48px; /* 增大图标 */
  color: #909399;
  margin-bottom: 10px;
}

.image-container p {
  font-size: 20px; /* 增大提示文字 */
  color: #606266;
}

.image-preview {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  padding: 10px;
  max-height: 400px;
  overflow-y: auto;
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

.add-more {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  aspect-ratio: 1;
  border: 2px dashed #dcdfe6;
  border-radius: 4px;
  cursor: pointer;
  background-color: #fafafa;
}

.add-more:hover {
  border-color: #409eff;
  background-color: #ecf5ff;
}

.add-more i {
  font-size: 30px;
  color: #909399;
  margin-bottom: 5px;
}

.add-more p {
  margin: 0;
  font-size: 14px;
  color: #606266;
}

.button-group {
  margin: 20px 0;
  text-align: center;
}

.button-group .el-button {
  padding: 12px 25px;
  font-size: 18px; /* 增大按钮文字 */
  font-weight: bold;
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
  font-size: 20px; /* 增大准确率文字 */
}

/* 自定义滚动条样式 */
.image-preview::-webkit-scrollbar {
  width: 6px;
}

.image-preview::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.image-preview::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}

.image-preview::-webkit-scrollbar-thumb:hover {
  background: #555;
}

/* Element UI 组件样式覆盖 */
:deep(.el-switch__label) {
  font-size: 24px !important; /* 增大开关文字 */
  font-weight: 500; /* 加粗文字 */
}

:deep(.el-switch__label.is-active) {
  color: #409eff !important;
  font-weight: bold; /* 激活状态更粗 */
}

/* 调整开关大小 */
:deep(.el-switch) {
  height: 32px; /* 增大开关高度 */
}

:deep(.el-switch__core) {
  height: 32px !important; /* 增大开关核心高度 */
  width: 60px !important; /* 增大开关宽度 */
}
</style>
