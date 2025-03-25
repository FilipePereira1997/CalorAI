# Etapa 1: build com Vite
FROM node:20 AS build

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm install

COPY . .

# Passa a variável para o build do Vite
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build

# Etapa 2: servir com nginx
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
