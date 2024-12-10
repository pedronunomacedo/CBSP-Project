# Use the official Node.js image as a base
FROM node:18

# Set the working directory in the container
WORKDIR /app

# Copy the package.json and package-lock.json files
COPY ./package.json ./package-lock.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY ./ ./

# Build the React app
RUN npm run build

# Install serve to serve the build
RUN npm install -g serve

# Expose port 3000 for the frontend
EXPOSE 3000

# Start the app with serve
CMD ["serve", "-s", "build", "-l", "3000"]
