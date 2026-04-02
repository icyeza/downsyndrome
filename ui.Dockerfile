# Use the lightweight Nginx image
FROM nginx:alpine

# Copy the static web files from the local ui folder to Nginx's serve directory
COPY ./ui /usr/share/nginx/html

# Copy your custom configuration to override the default Nginx settings
COPY ./nginx.conf /etc/nginx/conf.d/default.conf

# Nginx starts automatically, so no CMD is strictly needed, 
# but you can add it for clarity
CMD ["nginx", "-g", "daemon off;"]