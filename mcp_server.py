#!/usr/bin/env python3
"""
MCP (Minecraft Control Protocol) Server
A simple server implementation for controlling Minecraft servers remotely.
"""

import socket
import threading
import json
import logging
import time
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MCPServer")

class MCPServer:
    """
    MCP Server implementation that handles client connections and processes commands
    """
    def __init__(self, host='0.0.0.0', port=25575, max_connections=5, password=None):
        """
        Initialize the MCP server
        
        Args:
            host (str): The host address to bind to
            port (int): The port to listen on
            max_connections (int): Maximum number of concurrent connections
            password (str): Authentication password (optional)
        """
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.password = password
        self.server_socket = None
        self.clients = []
        self.running = False
        self.commands = {
            "help": self.cmd_help,
            "status": self.cmd_status,
            "list": self.cmd_list_players,
            "say": self.cmd_say,
            "stop": self.cmd_stop_server,
            "restart": self.cmd_restart_server,
            "execute": self.cmd_execute,
        }
        
        # Server state (simulated for now)
        self.server_state = {
            "online": True,
            "players": [],
            "max_players": 20,
            "version": "1.19.2",
            "motd": "MCP Controlled Server",
            "start_time": datetime.now(),
        }

    def start(self):
        """Start the MCP server and listen for connections"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_connections)
            self.running = True
            
            logger.info(f"MCP Server started on {self.host}:{self.port}")
            
            # Start accepting connections in a separate thread
            accept_thread = threading.Thread(target=self.accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
            # Main server loop
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Server shutdown requested via keyboard interrupt")
            finally:
                self.shutdown()
                
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            self.shutdown()

    def accept_connections(self):
        """Accept incoming client connections"""
        logger.info("Waiting for client connections...")
        
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                logger.info(f"New connection from {address[0]}:{address[1]}")
                
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
                self.clients.append({
                    "socket": client_socket,
                    "address": address,
                    "thread": client_thread,
                    "authenticated": False,
                    "connected_at": datetime.now()
                })
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting connection: {e}")
                    time.sleep(1)

    def handle_client(self, client_socket, address):
        """
        Handle communication with a connected client
        
        Args:
            client_socket: The client socket object
            address: The client address tuple (ip, port)
        """
        try:
            # Send welcome message
            self.send_response(client_socket, {
                "status": "success", 
                "message": "Welcome to MCP Server. Please authenticate."
            })
            
            authenticated = False
            if not self.password:
                authenticated = True
                logger.info(f"Client {address[0]}:{address[1]} automatically authenticated (no password set)")
                self.send_response(client_socket, {
                    "status": "success", 
                    "message": "Authentication successful"
                })
            
            while self.running:
                # Receive data from client
                data = client_socket.recv(4096)
                if not data:
                    logger.info(f"Client {address[0]}:{address[1]} disconnected")
                    break
                
                try:
                    request = json.loads(data.decode('utf-8'))
                    logger.debug(f"Received request: {request}")
                    
                    # Handle authentication
                    if not authenticated:
                        if "auth" in request and request["auth"] == self.password:
                            authenticated = True
                            logger.info(f"Client {address[0]}:{address[1]} authenticated successfully")
                            self.send_response(client_socket, {
                                "status": "success", 
                                "message": "Authentication successful"
                            })
                            
                            # Update client authentication status
                            for client in self.clients:
                                if client["socket"] == client_socket:
                                    client["authenticated"] = True
                                    break
                        else:
                            logger.warning(f"Failed authentication attempt from {address[0]}:{address[1]}")
                            self.send_response(client_socket, {
                                "status": "error", 
                                "message": "Authentication failed"
                            })
                            continue
                    
                    # Process command if authenticated
                    if authenticated and "command" in request:
                        self.process_command(client_socket, request)
                    
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from {address[0]}:{address[1]}")
                    self.send_response(client_socket, {
                        "status": "error", 
                        "message": "Invalid JSON format"
                    })
                
        except Exception as e:
            logger.error(f"Error handling client {address[0]}:{address[1]}: {e}")
        finally:
            # Clean up client connection
            try:
                client_socket.close()
                # Remove client from list
                self.clients = [c for c in self.clients if c["socket"] != client_socket]
                logger.info(f"Closed connection to {address[0]}:{address[1]}")
            except:
                pass

    def process_command(self, client_socket, request):
        """
        Process a command request from a client
        
        Args:
            client_socket: The client socket object
            request: The command request dictionary
        """
        command = request.get("command", "").lower()
        args = request.get("args", {})
        
        if command in self.commands:
            try:
                result = self.commands[command](args)
                self.send_response(client_socket, result)
            except Exception as e:
                logger.error(f"Error executing command '{command}': {e}")
                self.send_response(client_socket, {
                    "status": "error",
                    "message": f"Error executing command: {str(e)}"
                })
        else:
            logger.warning(f"Unknown command received: {command}")
            self.send_response(client_socket, {
                "status": "error",
                "message": f"Unknown command: {command}"
            })

    def send_response(self, client_socket, response):
        """
        Send a JSON response to a client
        
        Args:
            client_socket: The client socket object
            response: The response dictionary to send
        """
        try:
            json_response = json.dumps(response)
            client_socket.sendall(json_response.encode('utf-8'))
        except Exception as e:
            logger.error(f"Error sending response: {e}")

    def shutdown(self):
        """Shutdown the MCP server and clean up resources"""
        logger.info("Shutting down MCP server...")
        self.running = False
        
        # Close all client connections
        for client in self.clients:
            try:
                client["socket"].close()
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        logger.info("MCP server shutdown complete")

    # Command handlers
    def cmd_help(self, args):
        """Return help information about available commands"""
        commands_info = {
            "help": "Show available commands",
            "status": "Get server status information",
            "list": "List online players",
            "say": "Broadcast a message to all players",
            "stop": "Stop the Minecraft server",
            "restart": "Restart the Minecraft server",
            "execute": "Execute a command on the Minecraft server"
        }
        
        return {
            "status": "success",
            "message": "Available commands",
            "commands": commands_info
        }

    def cmd_status(self, args):
        """Return server status information"""
        uptime = datetime.now() - self.server_state["start_time"]
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        status = {
            "online": self.server_state["online"],
            "version": self.server_state["version"],
            "players_online": len(self.server_state["players"]),
            "max_players": self.server_state["max_players"],
            "motd": self.server_state["motd"],
            "uptime": uptime_str
        }
        
        return {
            "status": "success",
            "server_status": status
        }

    def cmd_list_players(self, args):
        """List online players"""
        return {
            "status": "success",
            "player_count": len(self.server_state["players"]),
            "max_players": self.server_state["max_players"],
            "players": self.server_state["players"]
        }

    def cmd_say(self, args):
        """Broadcast a message to all players"""
        message = args.get("message", "")
        if not message:
            return {
                "status": "error",
                "message": "No message provided"
            }
        
        logger.info(f"Broadcasting message: {message}")
        # In a real implementation, this would send the message to the Minecraft server
        
        return {
            "status": "success",
            "message": f"Message broadcast: {message}"
        }

    def cmd_stop_server(self, args):
        """Stop the Minecraft server"""
        logger.info("Stopping Minecraft server (simulated)")
        # In a real implementation, this would stop the Minecraft server
        self.server_state["online"] = False
        
        return {
            "status": "success",
            "message": "Minecraft server is stopping"
        }

    def cmd_restart_server(self, args):
        """Restart the Minecraft server"""
        logger.info("Restarting Minecraft server (simulated)")
        # In a real implementation, this would restart the Minecraft server
        self.server_state["online"] = True
        self.server_state["start_time"] = datetime.now()
        
        return {
            "status": "success",
            "message": "Minecraft server is restarting"
        }

    def cmd_execute(self, args):
        """Execute a command on the Minecraft server"""
        command = args.get("command", "")
        if not command:
            return {
                "status": "error",
                "message": "No command provided"
            }
        
        logger.info(f"Executing command: {command}")
        # In a real implementation, this would execute the command on the Minecraft server
        
        return {
            "status": "success",
            "message": f"Command executed: {command}",
            "result": "Command output would appear here"
        }


if __name__ == "__main__":
    # Get configuration from environment variables or use defaults
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "25575"))
    password = os.environ.get("MCP_PASSWORD", "")
    
    # Create and start the server
    server = MCPServer(host=host, port=port, password=password)
    server.start()
