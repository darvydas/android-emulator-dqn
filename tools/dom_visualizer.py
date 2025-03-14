import pygame
import time
import numpy as np
from lxml import etree

class DOMVisualizer:
    def __init__(self, android_env, width=1600, height=900):
        """Initialize the DOM tree visualizer with the Android environment."""
        self.env = android_env

        # Visualization properties
        self.width = width
        self.height = height
        self.background_color = (15, 15, 25)  # Darker background for better contrast
        self.node_radius = 20    # Slightly smaller nodes
        self.horizontal_spacing = 120  # Reduced horizontal spacing
        self.vertical_spacing = 70    # Reduced vertical spacing
        self.node_positions = {}  # Cache for node positions
        self.node_data = {}      # Store node data for all nodes

        # Node colors
        self.current_node_color = (255, 80, 80)    # Bright red for current node
        self.target_node_color = (80, 255, 80)     # Bright green for target node
        self.visited_node_color = (100, 180, 255)  # Blue for visited nodes
        self.regular_node_color = (200, 200, 200)  # Light gray for regular nodes
        self.node_border_color = (240, 240, 240)   # White border for all nodes

        # Animation control
        self.animation_speed = 1.0  # Seconds between steps
        self.last_action_time = time.time()
        self.pause_between_steps = True
        self.step_in_progress = False
        self.awaiting_next_step = False

        # Tree rendering and layout settings
        self.max_depth = 12      # Increased depth to show all nodes
        self.max_children = 10  # Maximum children to display per node
        self.use_full_tree = True  # Always draw full tree from root

        # Tracking
        self.visited_nodes = set()
        self.move_log = []  # Our own log history
        self.current_reward = 0
        self.total_reward = 0
        self.running = True
        self.last_target_xpath = None  # Track last target to detect episodes

        # Debug mode
        self.debug = True
        self.debug_messages = []

        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Android DOM Navigation Visualizer")

        # Fonts
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.node_font = pygame.font.SysFont('Arial', 14, bold=True)  # Reduced font size
        self.info_font = pygame.font.SysFont('Arial', 16)
        self.debug_font = pygame.font.SysFont('Arial', 14)

        # Setup clock for controlling framerate
        self.clock = pygame.time.Clock()

        print("DOM Visualizer initialized with improved display")

    def draw_node_explanation(self):
        """Draw explanation of the different node types and paths."""
        explanation_text = [
            "Red Nodes: Current position in the DOM tree",
            "Green Nodes: Target node to reach",
            "Blue Nodes: Previously visited nodes",
            "Red Lines: Path from root to current node",
            "Blue Dots: Path up from current to common ancestor",
            "Green Dots: Path down from common ancestor to target"
        ]

        y_pos = 60
        for text in explanation_text:
            text_surf = self.info_font.render(text, True, (200, 200, 200))
            self.screen.blit(text_surf, (20, y_pos))
            y_pos += 25

    def log_debug(self, message):
        """Add a debug message and print it."""
        if self.debug:
            print(f"DEBUG: {message}")
            self.debug_messages.append(message)
            if len(self.debug_messages) > 10:
                self.debug_messages = self.debug_messages[-10:]

    def check_episode_reset(self):
        """Check if we need to reset visited nodes for a new episode."""
        # Get current target
        current_target_xpath = getattr(self.env, '_target_node_xpath', None)

        # Check if agent reached target (current == target)
        current_node_xpath = getattr(self.env, '_current_node_xpath', None)

        # Reset condition: Target changed or current node reached target
        if current_target_xpath != self.last_target_xpath or current_node_xpath == current_target_xpath:
            if len(self.visited_nodes) > 0:
                self.log_debug("New episode detected - resetting visited nodes")
                self.visited_nodes.clear()

        # Update last target
        self.last_target_xpath = current_target_xpath

    def calculate_node_positions(self, root_node):
        """Calculate positions for the entire DOM tree visualization."""
        self.node_positions = {}
        self.node_data = {}

        # Check for episode reset
        self.check_episode_reset()

        # Get current and target xpath values directly from environment
        current_xpath = getattr(self.env, '_current_node_xpath', None)
        target_xpath = getattr(self.env, '_target_node_xpath', None)

        # Debug current and target
        if current_xpath:
            self.log_debug(f"Current node xpath from env: {current_xpath}")
        if target_xpath:
            self.log_debug(f"Target node xpath from env: {target_xpath}")

        # Get viewport dimensions for centering
        view_width = self.width - 350  # Reserve space for info panel

        # Track nodes by level and their parent relationships
        level_nodes = {}  # Dictionary of level -> list of nodes
        parent_map = {}   # Dictionary of node -> parent node
        sibling_order = {}  # Track order of siblings for each parent

        # First pass: Build level structure and parent relationships
        def process_node(node, parent=None, depth=0, sibling_index=0):
            if depth > self.max_depth:
                return

            # Store parent relationship
            if parent is not None:
                parent_xpath = self.env._get_xpath_from_xml_element(parent)
                if parent_xpath not in sibling_order:
                    sibling_order[parent_xpath] = []

                node_xpath = self.env._get_xpath_from_xml_element(node)
                sibling_order[parent_xpath].append(node_xpath)
                parent_map[node] = parent

            # Add node to level
            if depth not in level_nodes:
                level_nodes[depth] = []
            level_nodes[depth].append(node)

            # Process children
            children = list(node)
            if len(children) > self.max_children:
                children = children[:self.max_children]

            for i, child in enumerate(children):
                process_node(child, node, depth + 1, i)

        # Process the tree from root
        process_node(root_node)

        # Second pass: Position nodes by level
        for depth, nodes in level_nodes.items():
            # Calculate horizontal spacing
            total_width = len(nodes) * self.horizontal_spacing
            start_x = (view_width - total_width) / 2 + self.horizontal_spacing / 2

            for i, node in enumerate(nodes):
                # Calculate position
                x = start_x + i * self.horizontal_spacing
                y = 100 + depth * self.vertical_spacing

                # Get node xpath
                xpath = self.env._get_xpath_from_xml_element(node)

                # Determine node type
                is_current = (xpath == current_xpath)
                is_target = (xpath == target_xpath)
                is_visited = xpath in self.visited_nodes

                # Force update visited nodes
                if is_current:
                    self.visited_nodes.add(xpath)

                # Set the color and type
                if is_current:
                    node_type = "current"
                    color = self.current_node_color
                    self.log_debug(f"Current node found at {x},{y}: {xpath}")
                elif is_target:
                    node_type = "target"
                    color = self.target_node_color
                    self.log_debug(f"Target node found at {x},{y}: {xpath}")
                elif is_visited:
                    node_type = "visited"
                    color = self.visited_node_color
                else:
                    node_type = "regular"
                    color = self.regular_node_color

                # Get label text
                node_text = node.get('text', '')
                node_class = node.get('class', '').split('.')[-1] if node.get('class') else ''
                node_id = node.get('resource-id', '').split('/')[-1] if node.get('resource-id') else ''

                label = node_text or node_id or node_class or node.tag
                if len(label) > 12:
                    label = label[:10] + ".."

                # Store complete node data
                self.node_positions[xpath] = (x, y)
                self.node_data[xpath] = {
                    'node': node,
                    'parent': parent_map.get(node),
                    'type': node_type,
                    'color': color,
                    'label': label,
                    'xpath': xpath,
                    'siblings': sibling_order.get(parent_map.get(node, None) and
                                               self.env._get_xpath_from_xml_element(parent_map.get(node)), [])
                }

        return len(self.node_positions)

    def draw_tree_connections(self):
        """Draw all connections between nodes in a way that matches agent movement."""
        connections_drawn = 0

        # First, draw parent-to-first-child connections
        for xpath, data in self.node_data.items():
            if data['parent'] is None:
                continue  # Skip root node

            # Get parent xpath
            parent_xpath = self.env._get_xpath_from_xml_element(data['parent'])

            # Check if both nodes have positions
            if parent_xpath in self.node_positions and xpath in self.node_positions:
                # Check if this is the first child of the parent
                siblings = data.get('siblings', [])
                if siblings and siblings[0] == xpath:
                    child_x, child_y = self.node_positions[xpath]
                    parent_x, parent_y = self.node_positions[parent_xpath]

                    # Draw the parent-to-first-child connection
                    pygame.draw.line(
                        self.screen,
                        (100, 100, 120),  # Line color
                        (child_x, child_y - self.node_radius),  # Start at top of child
                        (parent_x, parent_y + self.node_radius),  # End at bottom of parent
                        2  # Line width
                    )
                    connections_drawn += 1

        # Then, draw connections between siblings (horizontal lines)
        for xpath, data in self.node_data.items():
            siblings = data.get('siblings', [])
            if len(siblings) <= 1:
                continue  # Skip if there are no siblings

            # Draw lines between adjacent siblings
            for i in range(len(siblings) - 1):
                sibling1 = siblings[i]
                sibling2 = siblings[i + 1]

                if sibling1 in self.node_positions and sibling2 in self.node_positions:
                    x1, y1 = self.node_positions[sibling1]
                    x2, y2 = self.node_positions[sibling2]

                    # Draw horizontal line between siblings
                    pygame.draw.line(
                        self.screen,
                        (100, 100, 120),  # Line color
                        (x1 + self.node_radius, y1),  # Right side of left sibling
                        (x2 - self.node_radius, y2),  # Left side of right sibling
                        2  # Line width
                    )
                    connections_drawn += 1

        self.log_debug(f"Drew {connections_drawn} connections")
        return connections_drawn

    def draw_tree_nodes(self):
        """Draw all nodes using the stored node data."""
        nodes_drawn = 0

        # Debug: Check what types we have
        current_count = sum(1 for data in self.node_data.values() if data['type'] == 'current')
        target_count = sum(1 for data in self.node_data.values() if data['type'] == 'target')
        visited_count = sum(1 for data in self.node_data.values() if data['type'] == 'visited')
        self.log_debug(f"Node types: {current_count} current, {target_count} target, {visited_count} visited")

        # Draw all nodes using our stored data
        for xpath, data in self.node_data.items():
            x, y = self.node_positions[xpath]
            color = data['color']
            node_type = data['type']
            label = data['label']

            # Draw the node circle
            pygame.draw.circle(self.screen, color, (x, y), self.node_radius)
            pygame.draw.circle(self.screen, self.node_border_color, (x, y), self.node_radius, 3)

            # Add glow effect for special nodes
            if node_type in ('current', 'target'):
                glow_radius = self.node_radius + 6
                pygame.draw.circle(self.screen, color, (x, y), glow_radius, 3)

            # Draw the label
            text_color = (0, 0, 0) if color != self.regular_node_color else (255, 255, 255)
            label_surf = self.node_font.render(label, True, text_color)
            label_rect = label_surf.get_rect(center=(x, y))
            self.screen.blit(label_surf, label_rect)

            nodes_drawn += 1

        return nodes_drawn

    def highlight_path_to_current(self):
        """Highlight the path from root to current node."""
        # Find the current node in our data
        current_xpath = None
        for xpath, data in self.node_data.items():
            if data['type'] == 'current':
                current_xpath = xpath
                break

        if not current_xpath:
            self.log_debug("No current node found for path highlighting")
            return

        self.log_debug(f"Highlighting path to current node: {current_xpath}")

        # Trace path up from current node to root
        path = []
        node_data = self.node_data[current_xpath]
        current_node = node_data['node']

        # Build path up to root
        current_xpath = self.env._get_xpath_from_xml_element(current_node)
        path.append(current_xpath)

        parent = node_data['parent']
        while parent is not None:
            parent_xpath = self.env._get_xpath_from_xml_element(parent)
            path.append(parent_xpath)

            # Get next parent
            if parent_xpath in self.node_data:
                parent = self.node_data[parent_xpath]['parent']
            else:
                break

        self.log_debug(f"Path to current has {len(path)} nodes")

        # Draw highlighted path
        for i in range(len(path) - 1):
            child_xpath = path[i]
            parent_xpath = path[i+1]

            if child_xpath in self.node_positions and parent_xpath in self.node_positions:
                child_x, child_y = self.node_positions[child_xpath]
                parent_x, parent_y = self.node_positions[parent_xpath]

                # Draw a thick red line
                pygame.draw.line(
                    self.screen,
                    (255, 100, 100),  # Bright red
                    (child_x, child_y - self.node_radius),
                    (parent_x, parent_y + self.node_radius),
                    4  # Thicker line
                )

    def highlight_path_to_target(self):
        """Highlight a path from current to target node."""
        # Find current and target nodes
        current_xpath = None
        target_xpath = None

        for xpath, data in self.node_data.items():
            if data['type'] == 'current':
                current_xpath = xpath
            elif data['type'] == 'target':
                target_xpath = xpath

        if not current_xpath or not target_xpath:
            self.log_debug(f"Missing nodes for path to target. Current: {current_xpath}, Target: {target_xpath}")
            return

        self.log_debug(f"Highlighting path from current to target: {current_xpath} -> {target_xpath}")

        # Get the nodes
        current_node = self.node_data[current_xpath]['node']
        target_node = self.node_data[target_xpath]['node']

        # Build path to root from current
        current_path = []
        node = current_node
        while node is not None:
            xpath = self.env._get_xpath_from_xml_element(node)
            current_path.append(xpath)

            # Get parent
            if xpath in self.node_data:
                node = self.node_data[xpath]['parent']
            else:
                break

        # Build path to root from target
        target_path = []
        node = target_node
        while node is not None:
            xpath = self.env._get_xpath_from_xml_element(node)
            target_path.append(xpath)

            # Get parent
            if xpath in self.node_data:
                node = self.node_data[xpath]['parent']
            else:
                break

        # Find common ancestor
        common_xpath = None
        for c_xpath in current_path:
            if c_xpath in target_path:
                common_xpath = c_xpath
                break

        if not common_xpath:
            self.log_debug("No common ancestor found")
            return

        self.log_debug(f"Common ancestor: {common_xpath}")

        # Get index of common ancestor in each path
        current_common_idx = current_path.index(common_xpath)
        target_common_idx = target_path.index(common_xpath)

        # Draw path from current to common ancestor with blue dots
        for i in range(current_common_idx):
            if i + 1 >= len(current_path):
                break

            child_xpath = current_path[i]
            parent_xpath = current_path[i+1]

            if child_xpath in self.node_positions and parent_xpath in self.node_positions:
                child_x, child_y = self.node_positions[child_xpath]
                parent_x, parent_y = self.node_positions[parent_xpath]

                # Draw blue dots
                for j in range(0, 10, 2):
                    fraction = j / 10
                    px = child_x + fraction * (parent_x - child_x)
                    py = child_y - self.node_radius + fraction * ((parent_y + self.node_radius) - (child_y - self.node_radius))
                    pygame.draw.circle(self.screen, (100, 180, 255), (px, py), 3)

        # Draw path from common ancestor to target with green dots
        for i in range(target_common_idx):
            if i + 1 >= len(target_path):
                break

            child_xpath = target_path[i]
            parent_xpath = target_path[i+1]

            if child_xpath in self.node_positions and parent_xpath in self.node_positions:
                child_x, child_y = self.node_positions[child_xpath]
                parent_x, parent_y = self.node_positions[parent_xpath]

                # Draw green dots
                for j in range(0, 10, 2):
                    fraction = j / 10
                    px = child_x + fraction * (parent_x - child_x)
                    py = child_y - self.node_radius + fraction * ((parent_y + self.node_radius) - (child_y - self.node_radius))
                    pygame.draw.circle(self.screen, (100, 255, 100), (px, py), 3)

    def draw_tree(self):
        """Draw the complete DOM tree."""
        # Clear the screen
        self.screen.fill(self.background_color)

        # Get current DOM tree
        root_node = self.env.get_root_node()

        if root_node is None:
            self.log_debug("DOM tree is None")
            self.draw_info_panel()
            pygame.display.flip()
            return

        # Draw explanation of node types and paths
        self.draw_node_explanation()

        # Calculate positions for all nodes and store data
        node_count = self.calculate_node_positions(root_node)
        self.log_debug(f"Positioned {node_count} nodes")

        # Draw connections first (lines between nodes)
        connections_drawn = self.draw_tree_connections()

        # Draw path from root to current node
        self.highlight_path_to_current()

        # Draw path from current to target if both exist
        self.highlight_path_to_target()

        # Draw all nodes (on top of connections)
        nodes_drawn = self.draw_tree_nodes()

        # Draw info panel and debugging information
        self.draw_info_panel()

        # Display animation controls if paused
        if self.awaiting_next_step:
            pause_text = "PAUSED - Press SPACE to step forward"
            pause_surf = self.title_font.render(pause_text, True, (255, 200, 100))
            self.screen.blit(pause_surf, (self.width // 2 - pause_surf.get_width() // 2, 10))

        # Add display info about the visualization
        info_text = f"Showing depth: {self.max_depth}, Nodes: {node_count}"
        info_surf = self.info_font.render(info_text, True, (180, 180, 180))
        self.screen.blit(info_surf, (10, 10))

        # Add controls help
        controls_text = "Controls: +/- (depth), P (toggle pause), SPACE (step), D (debug)"
        controls_surf = self.info_font.render(controls_text, True, (180, 180, 180))
        self.screen.blit(controls_surf, (10, 35))

        # Update the display
        pygame.display.flip()

    def draw_info_panel(self):
        """Draw the information panel showing agent status."""
        # Info panel background
        panel_rect = pygame.Rect(self.width - 350, 0, 350, self.height)
        pygame.draw.rect(self.screen, (40, 40, 45), panel_rect)
        pygame.draw.line(self.screen, (100, 100, 120), (self.width - 350, 0), (self.width - 350, self.height), 2)

        # Title
        title = self.title_font.render("Android DOM Navigation", True, (220, 220, 220))
        self.screen.blit(title, (self.width - 340, 20))

        # Current node info
        y_pos = 60
        self.screen.blit(self.info_font.render("Current Node:", True, (220, 220, 220)), (self.width - 340, y_pos))
        y_pos += 25

        if hasattr(self.env, '_current_node') and self.env._current_node is not None:
            # Get node details
            node_text = self.env._current_node.get('text', '')
            node_class = self.env._current_node.get('class', '')
            if node_class:
                node_class = node_class.split('.')[-1]

            node_resource_id = self.env._current_node.get('resource-id', '')
            if node_resource_id:
                node_resource_id = node_resource_id.split('/')[-1]

            if node_class:
                class_surf = self.info_font.render(f"Class: {node_class[:20]}", True, self.current_node_color)
                self.screen.blit(class_surf, (self.width - 340, y_pos))
                y_pos += 25

            if node_text:
                text_surf = self.info_font.render(f"Text: {node_text[:20]}", True, self.current_node_color)
                self.screen.blit(text_surf, (self.width - 340, y_pos))
                y_pos += 25

            if node_resource_id:
                id_surf = self.info_font.render(f"ID: {node_resource_id[:20]}", True, self.current_node_color)
                self.screen.blit(id_surf, (self.width - 340, y_pos))
                y_pos += 25

        # Target info
        y_pos += 20
        self.screen.blit(self.info_font.render("Target:", True, (220, 220, 220)), (self.width - 340, y_pos))
        y_pos += 25
        target_text = self.env._target_text if hasattr(self.env, '_target_text') else "None"
        target_surf = self.info_font.render(target_text, True, self.target_node_color)
        self.screen.blit(target_surf, (self.width - 340, y_pos))

        # Agent status
        y_pos += 40
        self.screen.blit(self.info_font.render("Agent Status:", True, (220, 220, 220)), (self.width - 340, y_pos))
        y_pos += 25

        reward_surf = self.info_font.render(f"Last Reward: {self.current_reward:.2f}", True,
                                          (100, 255, 100) if self.current_reward >= 0 else (255, 100, 100))
        self.screen.blit(reward_surf, (self.width - 340, y_pos))
        y_pos += 25

        total_surf = self.info_font.render(f"Total Reward: {self.total_reward:.2f}", True, (220, 220, 220))
        self.screen.blit(total_surf, (self.width - 340, y_pos))
        y_pos += 25

        # Recent actions
        y_pos += 20
        self.screen.blit(self.info_font.render("Recent Actions:", True, (220, 220, 220)), (self.width - 340, y_pos))
        y_pos += 25

        for msg in self.move_log[-8:]:  # Show last 8 log messages
            if "Moved" in msg:
                color = (180, 180, 255)  # Highlight movement actions
            elif "reward" in msg.lower():
                color = (255, 255, 180)  # Highlight rewards
            else:
                color = (200, 200, 200)

            # Truncate long messages
            if len(msg) > 40:
                msg = msg[:37] + "..."

            action_surf = self.info_font.render(msg, True, color)
            self.screen.blit(action_surf, (self.width - 340, y_pos))
            y_pos += 25

        # Draw legend at the bottom
        self.draw_legend()

    def draw_legend(self):
        """Draw a legend explaining node colors."""
        legend_items = [
            ("Current Node", self.current_node_color),
            ("Target Node", self.target_node_color),
            ("Visited Node", self.visited_node_color),
            ("Regular Node", self.regular_node_color)
        ]

        y_pos = self.height - 150
        self.screen.blit(self.info_font.render("Legend:", True, (220, 220, 220)), (self.width - 340, y_pos))
        y_pos += 30

        for text, color in legend_items:
            # Draw node style sample
            pygame.draw.circle(self.screen, color, (self.width - 320, y_pos), 10)
            pygame.draw.circle(self.screen, self.node_border_color, (self.width - 320, y_pos), 10, 2)

            # Draw label
            label = self.info_font.render(text, True, (220, 220, 220))
            self.screen.blit(label, (self.width - 295, y_pos - 8))
            y_pos += 30

    def log_action(self, message):
        """Add a message to our internal log history."""
        self.move_log.append(message)
        # Keep the log at a manageable size
        if len(self.move_log) > 100:
            self.move_log = self.move_log[-100:]

    def update_status(self, action=None, reward=None):
        """Update agent status information for visualization."""
        if reward is not None:
            # Check for target reached by large reward
            if reward > 10.0:
                self.log_debug(f"Large reward {reward} detected - possible target reached")
                # Will trigger reset on next draw

            self.current_reward = reward
            self.total_reward += reward
            self.log_action(f"Reward: {reward:.2f}")

        if action is not None:
            action_name = self.env.action_space[action]
            self.log_action(f"Action: {action_name}")

            # If the agent moved to a new node, log it
            if hasattr(self.env, '_current_node_xpath'):
                curr_xpath = self.env._current_node_xpath
                if curr_xpath:
                    short_xpath = curr_xpath.split('/')[-2:]
                    self.log_action(f"Node: .../{'/'.join(short_xpath)}")

            # Mark that a new action has been taken
            self.last_action_time = time.time()
            self.step_in_progress = True

            if self.pause_between_steps:
                self.awaiting_next_step = True

    def update_and_draw(self):
        """Update and draw the visualization - call this from your main loop."""
        # Handle PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN:
                # Add key controls for adjusting the visualization
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # Increase max depth
                    self.max_depth = min(self.max_depth + 1, 12)
                    self.log_debug(f"Increased depth to {self.max_depth}")
                elif event.key == pygame.K_MINUS:
                    # Decrease max depth
                    self.max_depth = max(self.max_depth - 1, 1)
                    self.log_debug(f"Decreased depth to {self.max_depth}")
                elif event.key == pygame.K_d:
                    # Toggle debug mode
                    self.debug = not self.debug
                    self.log_debug(f"Debug mode: {self.debug}")
                elif event.key == pygame.K_p:
                    # Toggle pause between steps
                    self.pause_between_steps = not self.pause_between_steps
                    self.log_debug(f"Pause between steps: {self.pause_between_steps}")
                    if not self.pause_between_steps:
                        self.awaiting_next_step = False
                elif event.key == pygame.K_SPACE:
                    # Continue to next step when paused
                    if self.awaiting_next_step:
                        self.awaiting_next_step = False
                        self.log_debug("Continuing to next step")
                elif event.key == pygame.K_r:
                    # Reset view
                    self.max_depth = 12
                    self.log_debug("View reset")
                elif event.key == pygame.K_c:
                    # Clear visited nodes manually
                    self.visited_nodes.clear()
                    self.log_debug("Manually cleared visited nodes")

        # Draw the tree
        self.draw_tree()

        # Cap the framerate
        self.clock.tick(30)  # Higher framerate for smoother animation

        # If we're waiting for the next step, return None to indicate waiting
        if self.awaiting_next_step:
            return None

        return True

    def close(self):
        """Clean up resources."""
        self.running = False
        pygame.quit()