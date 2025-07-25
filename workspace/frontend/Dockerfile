# frontend/Dockerfile
# React 환경을 위한 Dockerfile

# 1단계: 빌드 환경
FROM node:20-alpine AS build

WORKDIR /app
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile

COPY . .
RUN yarn build

# 2단계: Nginx을 이용한 정적 파일 서빙
FROM nginx:stable-alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
# React 라우팅을 위한 기본 설정 (옵션)
# COPY nginx.conf /etc/nginx/conf.d/default.conf
