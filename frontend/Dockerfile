# =============================================================================
# 前端 Docker 镜像：Vue3 + Vite（纯 Web）→ nginx 静态服务
#
# 多阶段构建：
#   Stage 1 (build)  : Node 20 执行 vite build，产物在 dist/
#   Stage 2 (runtime): nginx Alpine 提供静态文件 + /api 反向代理到 backend
# =============================================================================

# ---- 构建阶段 ----
FROM node:20-alpine AS build
WORKDIR /app

# npm 依赖层（源码未变时可缓存）
COPY package*.json ./
RUN npm install

# 源码 + 构建
COPY . .
RUN npm run build

# ---- 运行阶段 ----
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
