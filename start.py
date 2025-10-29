from deepface.api import app    # Import the ready-made DeepFace server
import os                        # Import tool to read environment variables

if __name__ == '__main__':      # This runs when you start the program
    port = int(os.environ.get('PORT', 5000))  # Get port number from Railway
    app.run(host='0.0.0.0', port=port)        # Start the server