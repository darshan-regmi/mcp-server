#!/usr/bin/env python3
"""
MCP Client
A simple client implementation to interact with the MCP server
"""

import socket
import json
import argparse
import sys

class MCPClient:
    """
    Client for connecting to and interacting with an MCP server
    """
    def __init__(self, host, port, password=None):
        """
        Initialize the MCP client
        
        Args:
            host (str): The MCP server host
            port (int): The MCP server port
            password (str): Authentication password (optional)
        """
        self.host = host
        self.port = port
        self.password = password
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to the MCP server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Receive welcome message
            response = self.receive_response()
            print(f"Server: {response.get('message', 'Connected to server')}")
            
            # Authenticate if password is set
            if self.password:
                auth_result = self.authenticate(self.password)
                if not auth_result:
                    print("Authentication failed.")
                    self.disconnect()
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error connecting to server: {e}")
            self.connected = False
            return False
            
    def authenticate(self, password):
        """
        Authenticate with the server
        
        Args:
            password (str): The authentication password
            
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        auth_request = {"auth": password}
        self.send_request(auth_request)
        response = self.receive_response()
        
        if response.get("status") == "success":
            print("Authentication successful")
            return True
        else:
            print(f"Authentication failed: {response.get('message', 'Unknown error')}")
            return False
            
    def send_request(self, request):
        """
        Send a request to the server
        
        Args:
            request (dict): The request to send
        """
        if not self.connected:
            print("Not connected to server")
            return
            
        try:
            json_request = json.dumps(request)
            self.socket.sendall(json_request.encode('utf-8'))
        except Exception as e:
            print(f"Error sending request: {e}")
            self.disconnect()
            
    def receive_response(self):
        """
        Receive a response from the server
        
        Returns:
            dict: The response from the server
        """
        if not self.connected:
            print("Not connected to server")
            return {}
            
        try:
            data = self.socket.recv(4096)
            if not data:
                print("Server closed connection")
                self.disconnect()
                return {}
                
            response = json.loads(data.decode('utf-8'))
            return response
        except Exception as e:
            print(f"Error receiving response: {e}")
            self.disconnect()
            return {}
            
    def disconnect(self):
        """Disconnect from the server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        print("Disconnected from server")
        
    def send_command(self, command, args=None):
        """
        Send a command to the server
        
        Args:
            command (str): The command to send
            args (dict): Command arguments
            
        Returns:
            dict: The server's response
        """
        if not args:
            args = {}
            
        request = {
            "command": command,
            "args": args
        }
        
        self.send_request(request)
        # Get the response from the server
        return self.receive_response()


def main():
    """Main function for the MCP client"""
    parser = argparse.ArgumentParser(description="MCP Client")
    parser.add_argument("--host", default="localhost", help="MCP server host")
    parser.add_argument("--port", type=int, default=25575, help="MCP server port")
    parser.add_argument("--password", help="MCP server password")
    parser.add_argument("--command", help="Run a single command and exit")
    parser.add_argument("--no-interactive", action="store_true", help="Don't use interactive mode even without a command")
    args = parser.parse_args()
    
    # Get password if not provided and we're not in non-interactive mode
    password = args.password
    if password is None and not args.no_interactive and not args.command:
        try:
            password = input("Enter server password (leave empty if none): ")
            if password == "":
                password = None
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user")
            sys.exit(1)
    
    # Create client and connect to server
    client = MCPClient(args.host, args.port, password)
    if not client.connect():
        sys.exit(1)
    
    try:
        if args.command:
            # Run a single command and exit
            command_parts = args.command.split(" ", 1)
            command = command_parts[0]
            
            command_args = {}
            if len(command_parts) > 1:
                if command == "say":
                    command_args = {"message": command_parts[1]}
                elif command == "execute":
                    command_args = {"command": command_parts[1]}
            
            response = client.send_command(command, command_args)
            print("\nCommand Response:")
            print(json.dumps(response, indent=2))
            
            # Display formatted response for better readability
            if response.get("status") == "success":
                if command == "help" and "commands" in response:
                    print("\nAvailable commands:")
                    for cmd, desc in response["commands"].items():
                        print(f"  {cmd}: {desc}")
                elif command == "status" and "server_status" in response:
                    status = response["server_status"]
                    print("\nServer Status:")
                    print(f"  Online: {status['online']}")
                    print(f"  Version: {status['version']}")
                    print(f"  Players: {status['players_online']}/{status['max_players']}")
                    print(f"  MOTD: {status['motd']}")
                    print(f"  Uptime: {status['uptime']}")
                elif command == "list" and "players" in response:
                    players = response["players"]
                    print(f"\nPlayers ({response['player_count']}/{response['max_players']}):")
                    if players:
                        for player in players:
                            print(f"  {player}")
                    else:
                        print("  No players online")
        else:
            # Interactive mode
            print("Connected to MCP server. Type 'help' for available commands, 'exit' to quit.")
            while client.connected:
                try:
                    cmd_input = input("MCP> ")
                    if not cmd_input:
                        continue
                    
                    if cmd_input.lower() == "exit" or cmd_input.lower() == "quit":
                        break
                    
                    # Parse command and arguments
                    parts = cmd_input.split(" ", 1)
                    command = parts[0].lower()
                    
                    command_args = {}
                    if len(parts) > 1:
                        if command == "say":
                            command_args = {"message": parts[1]}
                        elif command == "execute":
                            command_args = {"command": parts[1]}
                    
                    # Send command to server
                    response = client.send_command(command, command_args)
                    
                    # Display response
                    if response.get("status") == "success":
                        if "message" in response:
                            print(f"Success: {response['message']}")
                        
                        # Display additional information based on command
                        if command == "help" and "commands" in response:
                            print("\nAvailable commands:")
                            for cmd, desc in response["commands"].items():
                                print(f"  {cmd}: {desc}")
                        elif command == "status" and "server_status" in response:
                            status = response["server_status"]
                            print("\nServer Status:")
                            print(f"  Online: {status['online']}")
                            print(f"  Version: {status['version']}")
                            print(f"  Players: {status['players_online']}/{status['max_players']}")
                            print(f"  MOTD: {status['motd']}")
                            print(f"  Uptime: {status['uptime']}")
                        elif command == "list" and "players" in response:
                            players = response["players"]
                            print(f"\nPlayers ({response['player_count']}/{response['max_players']}):")
                            if players:
                                for player in players:
                                    print(f"  {player}")
                            else:
                                print("  No players online")
                    else:
                        print(f"Error: {response.get('message', 'Unknown error')}")
                
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user")
                    break
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error: {e}")
    
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
