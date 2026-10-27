# Stage 1: Build React bundle using Node
FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Stage 2: Serve compiled static assets via lightweight Nginx
FROM nginx:alpine

# Copy built dist files to default Nginx web root
COPY --from=build /app/dist /usr/share/nginx/html

# Expose Vite/Web port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
