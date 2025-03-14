# android_env.py
import random
import time
from appium import webdriver
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.common.actions import interaction
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_actions import PointerActions
from appium.options.android import UiAutomator2Options
from selenium.common.exceptions import NoSuchElementException
import numpy as np
import tensorflow as tf
from lxml import etree
from tools.bert import tokenize_string

class AndroidEnv:
    def __init__(self, desired_caps, logger, target_apps=None):
        """
        Initializes the AndroidEnv environment with expanded actions and Accessibility Tree state.
        """
        self.logger = logger

        options = UiAutomator2Options()
        options.platform_name = desired_caps.get('platformName')
        options.device_name = desired_caps.get('deviceName')
        options.platform_version = desired_caps.get('platformVersion')
        options.app_package = desired_caps.get('appPackage')
        options.app_activity = desired_caps.get('appActivity')
        options.automation_name = desired_caps.get('automationName')
        options.new_command_timeout = desired_caps.get('newCommandTimeout')
        if 'chromedriverExecutable' in desired_caps:
            options.chromedriver_executable = desired_caps.get('chromedriverExecutable')
        options.no_reset = True

        self.driver = webdriver.Remote(command_executor='http://127.0.0.1:4723', options=options)
        self.action_space = [
            "move_to_child",
            "move_to_next_sibling",
            "move_to_previous_sibling",
            "move_to_parent"
            # "tap_element",
            # "scroll_down",
            # "scroll_up",
            # "scroll_left",
            # "scroll_right",
            # "input_text",
            # "press_back",
            # "press_home",
            # "press_menu",
            # "noop"
        ]
        self.action_size = len(self.action_space)
        self.state_size = 54
        self.messaging_app_package = "com.google.android.apps.messaging"
        self.messaging_app_activity = "com.android.messaging.ui.conversationlist.ConversationListActivity"
        self.settings_app_package = "com.android.settings"
        self.max_steps_per_episode = 300
        self.pointer = PointerInput(interaction.POINTER_TOUCH, "touch")
        self.action_builder = ActionBuilder(self.driver, duration=50)
        self._current_node = None
        self._current_node_xpath = None
        self._target_apps = target_apps or ["Messages"]  # Default to Messages if none provided
        self._target_text = None
        self._target_node = None
        self._target_node_xpath = None


        self.class_encodings = set()
        self.resource_id_encodings = set()

    def reset(self, target_app=None):
        """Resets the environment and returns initial state (Accessibility Tree)."""
        self.driver.press_keycode(3)  # HOME key
        time.sleep(1)
        self.current_step = 0

        # Set target app
        if target_app is None and self._target_apps:
            self._target_text = random.choice(self._target_apps)

        root_node = self.get_root_node()

        self._current_node = root_node
        # move cursor next to
        # self._move_to_child(0)
        # self._move_to_child(0)
        # self._move_to_child(0)
        # self._move_to_child(0)
        # self._move_to_child(0)
        # self._move_to_child(4)
        # self._move_to_child(0)
        # self._move_to_child(0) #if moves to next sibling - should reach done
        self._current_node_xpath = self._get_xpath_from_xml_element(self._current_node)
        self.logger.append_move_log(f'Current node: {self._current_node_xpath}')

        target_node_search_result = root_node.xpath(f'//*[@text="{self._target_text}"]')
        if len(target_node_search_result) > 0:
            self._target_node = target_node_search_result[0]
            self._target_node_xpath = self._get_xpath_from_xml_element(self._target_node)
            self.logger.append_move_log(f'Target node: {self._target_node_xpath}')
        else:
            self._target_node = None
            self._target_node_xpath = None
            self.logger.append_move_log(f"Target {self._target_text} not found in current view")

        return self.get_state()

    def get_root_node(self):
        """Returns the root node of the current Accessibility Tree."""
        accessibility_tree_xml = self.get_state_xml()
        root = etree.fromstring(accessibility_tree_xml.encode('utf-8'))
        return root

    def get_state_xml(self):
        """Returns current state as Accessibility Tree (page source in XML)."""
        return self.driver.page_source


    def get_state(self):
        """Returns current state as features extracted from the current node's Accessibility Tree."""
        if self._current_node is not None:
            return self.extract_features_from_tree(self._current_node) # Extract features for only the current node
        else:
            return np.zeros(8 * 1, dtype=np.float32) # Return zero vector if current_node is None

    def _get_xpath_from_xml_element(self, xml_element):
        """
        Helper function to generate a simplified XPath from an XML element.
        This is a basic implementation and might need adjustments for complex UIs.
        """
        path_components = []
        current = xml_element
        while current is not None and current.tag != 'hierarchy': # Stop at the root
            component = current.tag
            if 'index' in current.attrib: # Basic index-based child selection
                component += f"[{int(current.attrib['index']) + 1}]" # XPath is 1-indexed
            path_components.append(component)
            current = current.getparent()

        return "/" + "/".join(reversed(path_components)) # Construct XPath from root to element

    def close(self):
        """Quits Appium driver."""
        self.driver.quit()

    def get_available_actions(self):
        """Returns a list of available actions."""
        available_action_space = []
        for index, action_name in enumerate(self.action_space):
            if self._current_node is not None:
                if action_name == "move_to_child":
                    # if current_node has children return index, else skip
                    if len(list(self._current_node)) > 0:
                        available_action_space.append(index)
                elif action_name == "move_to_next_sibling":
                    if self._current_node.getnext() is not None:
                        available_action_space.append(index)
                elif action_name == "move_to_previous_sibling":
                    if self._current_node.getprevious() is not None:
                        available_action_space.append(index)
                elif action_name == "move_to_parent":
                    if self._current_node.getparent() is not None and self._current_node.getparent().tag != 'hierarchy':
                        available_action_space.append(index)
                elif action_name == "noop":
                    available_action_space.append(index)

        self.logger.append_move_log(f'Available actions: {available_action_space}')

        return available_action_space

    def step(self, action_index, action_params=None):
        """Executes action, returns next state (Accessibility Tree), reward, done, info."""
        action_name = self.action_space[action_index]
        reward = -0.1
        done = False
        info = {}
        error = False

        try:
            if action_name == "move_to_child":
                reward += self._move_to_child(0)
                # if action_params and 'index' in action_params:
                #     self._move_to_child(action_params['index'])
            elif action_name == "move_to_next_sibling":
                reward += self._move_to_sibling('next')
            elif action_name == "move_to_previous_sibling":
                reward += self._move_to_sibling('previous')
            elif action_name == "move_to_parent":
                reward += self._move_to_parent()
            # elif action_name == "tap_element":
            #     if action_params and 'locator' in action_params:
            #         self._tap_element(action_params['locator'])
            #     else:
            #         reward = -5 # Penalty for missing parameters
            #         error = True
            #         print("Error: tap_element action requires 'locator' parameter.")
            # elif action_name == "scroll_down":
            #     self._scroll(direction="down")
            # elif action_name == "scroll_up":
            #     self._scroll(direction="up")
            # elif action_name == "scroll_left":
            #     self._scroll(direction="left")
            # elif action_name == "scroll_right":
            #     self._scroll(direction="right")
            # elif action_name == "input_text":
            #     if action_params and 'locator' in action_params and 'text_to_input' in action_params:
            #         self._input_text(action_params['locator'], action_params['text_to_input'])
            #     else:
            #         reward = -5
            #         error = True
            #         print("Error: input_text action requires 'locator' and 'text_to_input' parameters.")
            # elif action_name == "press_back":
            #     self._press_system_button(button_type="back")
            # elif action_name == "press_home":
            #     self._press_system_button(button_type="home")
            # elif action_name == "press_menu":
            #     self._press_system_button(button_type="menu")
            elif action_name == "noop":
                # reward = 0
                self.logger.append_move_log(f'No operation')


            else:
                reward = -10
                error = True
                print(f"Error: Unknown action: {action_name}")

            next_state = self.get_state()
            # self.logger.append_move_log(f'New state features: {next_state}')

            if 'text' in self._current_node.attrib.keys() and self._current_node.attrib['text'] == self._target_text:
                reward = 200
                done = True
                # print(f'DONE x {self._current_node_xpath}')
                self.logger.append_move_log(f'TARGET REACHED: {self._current_node_xpath}')
            elif self.driver.current_package == self.messaging_app_package:
                reward = 100
                done = True
            elif self.driver.current_package == self.settings_app_package:
                reward = -5
                done = False

            # Apply similarity reward if not done
            if not done and self._current_node is not None and self._target_node is not None:
                similarity_reward = self.calculate_xpath_similarity_reward()
                reward += similarity_reward

                # Log the current state and reward
                self.logger.append_move_log(f"Current: {self._current_node_xpath}")
                self.logger.append_move_log(f"Target: {self._target_node_xpath}")
                self.logger.append_move_log(f"Total reward: {reward}")


            self.current_step += 1
            if self.current_step >= self.max_steps_per_episode:
                done = True

        except Exception as e:
            print(f"Error executing action: {action_name}, Error: {e}")
            reward = -10
            next_state = self.get_state()
            done = True
            error = True

        return next_state, reward, done, info

    def _move_to_child(self, index):
        """Moves the current node to the child at the given index."""
        if self._current_node is not None:
            children = list(self._current_node)
            if 0 <= index < len(children):
                self._current_node = children[index]
                self._current_node_xpath = self._get_xpath_from_xml_element(self._current_node) # Update XPath
                # print(f"Moved ↓ {self._current_node_xpath}")
                self.logger.append_move_log(f"Moved ↓ {self._current_node_xpath}")

                reward = 0
            else:
                print(f"Invalid child index {index}. Current node has {len(children)} children.")
                reward = -1
        else:
            print("No current node to move from.")
            reward = -5

        return reward

    def _move_to_sibling(self, direction):
        """Moves the current node to the next or previous sibling."""
        if self._current_node is None:
            print("No current node to move from.")
            return -5 # reward

        if direction == 'next':
            sibling = self._current_node.getnext()
            if sibling is None:
                print("Current node has no next siblings.")
                return -1 # reward

            self._current_node = sibling
            self._current_node_xpath = self._get_xpath_from_xml_element(self._current_node) # Update XPath
            # print(f"Moved → {self._current_node_xpath}")
            self.logger.append_move_log(f"Moved → {self._current_node_xpath}")
            return 0 # reward

        elif direction == 'previous':
            sibling = self._current_node.getprevious()
            if sibling is None:
                print("Current node has no previous siblings.")
                return -1 # reward

            self._current_node = sibling
            self._current_node_xpath = self._get_xpath_from_xml_element(self._current_node) # Update XPath
            # print(f"Moved ← {self._current_node_xpath}")
            self.logger.append_move_log(f"Moved ← {self._current_node_xpath}")
            return 0 # reward

        else:
            print(f"Invalid sibling direction: {direction}. Use 'next' or 'previous'.")
            return -5 # reward

    def _move_to_parent(self):
        """Moves the current node to its parent."""
        if self._current_node is not None:
            parent = self._current_node.getparent()
            if parent is not None and parent.tag != 'hierarchy': # Ensure not moving beyond root
                self._current_node = parent
                self._current_node_xpath = self._get_xpath_from_xml_element(self._current_node)
                # print(f"Moved ↑ {self._current_node_xpath}")
                self.logger.append_move_log(f"Moved ↑ {self._current_node_xpath}")
                return 0 # reward
            else:
                print("Current node is root or has no parent.")
                return -1 # reward
        else:
            print("No current node to move from.")
            return -5 # reward

    def extract_features_from_tree(self, current_node_xml):
        """
        Extracts features from an Accessibility Tree (XML format from Appium) and returns a fixed-size numerical vector.

        Args:
            accessibility_tree_xml (str): The Accessibility Tree in XML format as a string (from Appium's page_source).

        Returns:
            np.ndarray: Fixed-size numerical state vector.
        """
        if current_node_xml is None:
            return np.zeros(self.state_size, dtype=np.float32)

        node_features = []
        # Feature Extraction from XML Element Attributes
        # isTopElement: 0, get xpath depth from root?

        # 1. Element Class Encoding
        class_name = current_node_xml.get('class')
        node_features.extend(tokenize_string(class_name, max_length=12) if class_name else np.zeros(12, dtype=np.float32))

        # isTextNodeVisible: 0, text != \n,"",None?
        # 2. Text Content Presence (Binary)
        text_content = current_node_xml.get('text')
        node_features.extend(tokenize_string(text_content, max_length=12) if text_content else np.zeros(12, dtype=np.float32))

        # 3. Resource ID Encoding
        resource_id = current_node_xml.get('resource-id')  # Note: attribute name is 'resource-id'
        if resource_id:
            resource_app = resource_id.split(':')[0]
            resource_action = resource_id.split(':')[1]
            node_features.extend(tokenize_string(resource_app, max_length=12) if resource_app else np.zeros(12, dtype=np.float32))
            node_features.extend(tokenize_string(resource_action, max_length=12) if resource_action else np.zeros(12, dtype=np.float32))
        else:
            node_features.extend(np.zeros(24, dtype=np.float32))

        # isInteractiveElement: 0,
        clickable = current_node_xml.get('clickable') == 'true'
        # isElementVisible: 0,
        focusable = current_node_xml.get('focusable') == 'true'
        enabled = current_node_xml.get('enabled') == 'true'

        checked = current_node_xml.get('checked') == 'true'
        selected = current_node_xml.get('selected') == 'true'
        # 4-8. Element States (Clickable, Focusable, Enabled, Checked, Selected)
        node_features.extend([int(clickable), int(focusable), int(enabled), int(checked), int(selected)])

        # isInExpandedViewport: 0,
        # getEffectiveScroll: 0,

        xpath = self._get_xpath_from_xml_element(current_node_xml)
        xpath_depth = xpath.count('/') - 1 # Count '/' separators, subtract 1 for root '/'
        node_features.append(xpath_depth) # Add XPath depth as a feature

        self.logger.append_move_log(f"Features: {node_features}")
        return np.array(node_features)

    def calculate_xpath_similarity_reward(self):
        """
        Calculate reward based on XPath similarity and node attributes
        """
        # Split paths into components
        current_components = self._current_node_xpath.split('/')
        target_components = self._target_node_xpath.split('/')

        # Calculate path similarity
        common_depth = 0
        for i in range(min(len(current_components), len(target_components))):
            if current_components[i] == target_components[i]:
                common_depth += 1
            else:
                break

        # Base reward for path similarity
        max_depth = max(len(current_components), len(target_components))
        path_reward = 2.0 * common_depth / max_depth  # Scaled between 0-2

        # Distance penalty - penalize being far from the target in tree
        remaining_distance = len(target_components) - common_depth
        distance_penalty = -0.1 * remaining_distance  # Small penalty for being far

        total_reward = path_reward + distance_penalty

        # Add debugging info
        self.logger.append_move_log(f"Similarity reward: {total_reward:.2f} (Path: {path_reward:.2f}, Dist: {distance_penalty:.2f})")

        return total_reward

    def get_bounds(self, xml_element):
        """Gets the bounds (rect) of an element from its XML representation."""
        bounds_str = xml_element.get('bounds')
        if bounds_str:
            bounds = bounds_str.split('][')
            x1, y1 = map(int, bounds[0][1:].split(','))
            x2, y2 = map(int, bounds[1][:-1].split(','))
            return {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}
        return None

    def draw_rectangle(self, x, y, width, height, color="red"):
        """Draws a rectangle on the device screen using JavaScript."""
        # Switch to WebView context
        original_context = self.driver.current_context
        self.driver.switch_to.context(self.driver.contexts[1])

        js_code = f"""
            var canvas = document.createElement('canvas');
            canvas.style.position = 'absolute';
            canvas.style.top = '0px';
            canvas.style.left = '0px';
            canvas.style.zIndex = '10000';
            canvas.style.pointerEvents = 'none';
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            document.body.appendChild(canvas);
            var ctx = canvas.getContext('2d');
            ctx.strokeStyle = '{color}';
            ctx.lineWidth = 2;
            ctx.strokeRect({x}, {y}, {width}, {height});
        """
        self.driver.execute_script(js_code)

        # Switch back to the original context
        self.driver.switch_to.context(original_context)

    def _scroll_current_element(self, direction="down", duration_ms=200, distance_px=300):
        """Scrolls the current UI element in a given direction."""

        current_element_xml = self._current_node  # Assuming self._current_node stores the XML element
        if current_element_xml is None:
            print("Warning: No current element to scroll.")
            return

        locator = self._get_locator_from_xml_element(current_element_xml)
        element = self._find_element_in_current_node(locator) # Use _find_element_in_current_node
        if not element:
            print(f"Warning: Current element not found for scroll action with locator: {locator} within current node.")
            return

        if element:
            rect = element.rect # Get element bounds
            start_x = rect['x'] + rect['width'] // 2
            start_y = rect['y'] + rect['height'] // 2
            end_x, end_y = start_x, start_y

            if direction == "down":
                end_y = start_y - distance_px
            elif direction == "up":
                end_y = start_y + distance_px
            elif direction == "left":
                end_x = start_x + distance_px
            elif direction == "right":
                end_x = start_x - distance_px
            else:
                raise ValueError(f"Invalid scroll direction: {direction}")

            # Ensure end coordinates are within bounds of the screen (not element, for scrolling effect)
            window_size = self.driver.get_window_size()
            width = window_size['width']
            height = window_size['height']
            end_x = max(0, min(end_x, width))
            end_y = max(0, min(end_y, height))

            self.action_builder.pointer_action.move_to_location(start_x, start_y)
            self.action_builder.pointer_action.pointer_down()
            self.action_builder.pointer_action.pause(0.1)
            self.action_builder.pointer_action.move_to_location(end_x, end_y)
            self.action_builder.pointer_action.release()
            self.action_builder.perform()

            time.sleep(0.5)
        else:
            print(f"Warning: Current element not found for scroll action.")

    def _get_locator_from_xml_element(self, xml_element):
        """Helper function to create a locator from an XML element."""
        resource_id = xml_element.get('resource-id')
        text = xml_element.get('text')
        class_name = xml_element.get('class')
        xpath = self._get_xpath_from_xml_element(xml_element) # Implement xpath generation if needed

        locator = {}
        if resource_id:
            locator['resource_id'] = resource_id
        elif text:
            locator['text'] = text
        elif class_name:
            locator['class_name'] = class_name
        elif xpath:
            locator['xpath'] = xpath # Fallback to xpath if other locators are missing
        else:
            return None # No locator info

        return locator




    def _tap_element(self, locator):
        """Taps on a UI element based on locator."""
        element = self._find_element_in_current_node(locator)
        if element:
            element.click()
            time.sleep(0.5)
        else:
            print(f"Warning: Element not found for tap action with locator: {locator}")

    def _input_text(self, locator, text_to_input):
        """Inputs text into a text field."""
        element = self._find_element_in_current_node(locator)
        if element:
            element.send_keys(text_to_input)
            time.sleep(0.5)
        else:
            print(f"Warning: Element not found for input_text action with locator: {locator}")

    def _scroll(self, direction="down", duration_ms=200, distance_px=500):
        """Scrolls in a given direction using pointer actions."""
        # Scrolls the whole screen
        window_size = self.driver.get_window_size()
        start_x = window_size['width'] // 2
        start_y = window_size['height'] // 2
        end_x, end_y = start_x, start_y

        if direction == "down":
            end_y = start_y - distance_px
        elif direction == "up":
            end_y = start_y + distance_px
        elif direction == "left":
            end_x = start_x + distance_px
        elif direction == "right":
            end_x = start_x - distance_px
        else:
            raise ValueError(f"Invalid scroll direction: {direction}")

        self.action_builder.pointer_action.move_to_location(start_x, start_y)
        self.action_builder.pointer_action.pointer_down()
        self.action_builder.pointer_action.pause(0.1)
        self.action_builder.pointer_action.move_to_location(end_x, end_y)
        self.action_builder.pointer_action.release()
        self.action_builder.perform()

        time.sleep(0.5)

    def _press_system_button(self, button_type):
        """Presses a system button using keycode."""
        if button_type == "back":
            self.driver.press_keycode(4)  # Keycode for BACK
        elif button_type == "home":
            self.driver.press_keycode(3)  # Keycode for HOME
        elif button_type == "menu":
            self.driver.press_keycode(82) # Keycode for MENU
        time.sleep(0.5)

    def get_text_of_element(self, locator):
        """Retrieves text content of a UI element."""
        element = self._find_element(locator)
        if element:
            return element.text
        else:
            print(f"Warning: Element not found for get_text_of_element with locator: {locator}")
            return None

    def get_element_attributes(self, locator, attributes_list):
        """Retrieves attributes of a UI element."""
        element = self._find_element(locator)
        if element:
            attributes = {}
            for attr_name in attributes_list:
                attributes[attr_name] = element.get_attribute(attr_name)
            return attributes
        else:
            print(f"Warning: Element not found for get_element_attributes with locator: {locator}")
            return None

    def _find_element(self, locator):
        """Finds a UI element based on the provided locator dictionary."""
        try:
            if 'resource_id' in locator:
                return self.driver.find_element(by=AppiumBy.ID, value=locator['resource_id'])
            elif 'text' in locator:
                return self.driver.find_element(by=AppiumBy.TEXT, value=locator['text'])
            elif 'class_name' in locator:
                return self.driver.find_element(by=AppiumBy.CLASS_NAME, value=locator['class_name'])
            elif 'xpath' in locator: # Added xpath locator
                return self.driver.find_element(by=AppiumBy.XPATH, value=locator['xpath'])
            else:
                print(f"Error: Unsupported locator type in: {locator}")
                return None
        except NoSuchElementException:
            print(f"Element not found with locator: {locator}")
            return None
        except Exception as e:
            print(f"Error finding element with locator: {locator}, Error: {e}")
            return None

    def _find_element_in_current_node(self, locator):
        """Finds a UI element within the subtree of the current node."""
        if self._current_node is None:
            print("Warning: No current node set, cannot search within it.")
            return None

        try:
            if 'resource_id' in locator:
                return self._current_node.find_element(by=AppiumBy.ID, value=locator['resource_id'])
            elif 'text' in locator:
                return self._current_node.find_element(by=AppiumBy.TEXT, value=locator['text'])
            elif 'class_name' in locator:
                return self._current_node.find_element(by=AppiumBy.CLASS_NAME, value=locator['class_name'])
            elif 'xpath' in locator:
                return self._current_node.find_element(by=AppiumBy.XPATH, value=locator['xpath'])
            else:
                print(f"Error: Unsupported locator type in: {locator}")
                return None

        except NoSuchElementException:
            print(f"Element not found with locator: {locator} within current node.")
            return None
        except Exception as e:
            print(f"Error finding element with locator: {locator} within current node, Error: {e}")
            return None





# DQN Model Placeholder
def create_dqn_model(input_shape, num_actions):
    """Placeholder for a simple DQN model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == "__main__":
    # Appium Desired Capabilities (adjust to your emulator setup)
    desired_caps = {
        "platformName": "Android",
        "platformVersion": "15.0",  # Adjust to your emulator version
        "deviceName": "Pixel_8_API_Vanilla_Hardware",  # Or the name of your emulator
        "appPackage": "com.google.android.apps.nexuslauncher",  # Default launcher package
        "appActivity": "com.google.android.apps.nexuslauncher.NexusLauncherActivity",  # Default launcher activity
        "automationName": "UiAutomator2",
        "newCommandTimeout": 100,
        'chromedriverExecutable': '/path/to/chromedriver'  # Optional: path to chromedriver if needed
    }

    # # Initialize environment
    # env = AndroidEnv(desired_caps)
    # env.reset()
    # action_size = env.action_size

    # # Example Usage
    # # # 1. Get Accessibility Tree
    # # accessibility_tree_xml = env.get_state_xml()
    # # print("Sample Accessibility Tree (XML):\n", accessibility_tree_xml[:500], "...")  # Print first 500 chars

    # # 2. Extract Features from Tree
    # # state_vector = extract_features_from_tree(accessibility_tree_xml)
    # state_vector = env.get_state()
    # print("\nState Vector Shape:", state_vector.shape)
    # print("Sample State Vector:", state_vector[:50])  # Print first 50 elements

    # # 3. Create DQN Model
    # state_size = state_vector.shape
    # dqn_model = create_dqn_model(state_size, action_size)
    # dqn_model.summary()

    # # 4. Example Action Calls
    # # Tap on a UI element
    # locator = {"text": "Messages"}
    # env._tap_element(locator)
    # time.sleep(2)

    # #scroll down
    # env._scroll("down")
    # time.sleep(2)

    # #input text
    # locator_text = {"class_name" : "android.widget.EditText"}
    # text_to_input = "Hello"
    # env._input_text(locator_text,text_to_input)
    # time.sleep(2)

    # #press back
    # env._press_system_button("back")
    # time.sleep(2)

    # Initialize environment
    env = AndroidEnv(desired_caps)
    action_size = env.action_size

    # Example Usage
    # 1. Get Accessibility Tree and initial state
    state_vector = env.reset()
    print("\nInitial State Vector Shape:", state_vector.shape)
    print("Sample Initial State Vector:", state_vector[:50])

    # accessibility_tree_xml = env.get_state_xml()
    # print("Sample Accessibility Tree (XML):\n", accessibility_tree_xml[:500], "...")
    # root_tree = etree.fromstring(accessibility_tree_xml.encode('utf-8'))
    # for element in root_tree.iter():
    #     print(f"{element.tag} - {element.text}")
    # print(root_tree[0])
    # print(root_tree[1])
    # print(root_tree[2])
    # print(root_tree[3])
    # print(root_tree[4])
    # print(root_tree[5])



    # 2. Demonstrate Navigation Actions
    print("\nDemonstrating Navigation Actions:")

    print("\nCurrent Node (Initial):", env._current_node.tag) # Should be 'hierarchy'

    env._move_to_child(0) # Move to the first child of root
    print("Current Node (Child 0):", env._current_node.tag)

    env._move_to_child(0) # Move to the first child of the new current node
    print("Current Node (Child 0 of Child 0):", env._current_node.tag)

    print(type(env._current_node))

    env._move_to_parent()
    print("Current Node (Parent):", env._current_node.tag) # Should be back to the first child of root


    env._move_to_child(0)
    env._move_to_child(0)
    env._move_to_child(0)
    env._move_to_child(0)
    env._move_to_child(4)
    env._move_to_child(0)
    env._move_to_child(1)
    print("Current Node (With Sibling):", env._current_node.tag)
    if 'text' in env._current_node.attrib and env._current_node.attrib['text'] == "Messages":
        print('yes, founded')
    print(env._current_node.attrib)
    print(env._current_node.attrib['text'])
    print(env._current_node)

    env._move_to_sibling('next')
    print("Current Node (Next Sibling):", env._current_node.tag)
    if env._current_node.attrib['text'] == "Messages":
        print('should not find')
    env._move_to_child(0)

    bounds = env.get_bounds(env._current_node)
    print(bounds)
    env.draw_rectangle(bounds['x'], bounds['y'], bounds['width'], bounds['height'])
    time.sleep(2)



    env._move_to_sibling('previous')
    print("Current Node (Previous Sibling):", env._current_node.tag) # Should move back to the first child of root

    # 3. Demonstrate Interaction Actions (operating on current_node subtree)
    print("\nDemonstrating Interaction Actions on Current Node Subtree:")

    # Assuming current node is root, try to tap "Messages" - might need to adjust based on your emulator's UI
    locator_tap = {"text": "Messages"} # Or adjust locator to something visible from root
    print("Attempting to tap 'Messages' from current node subtree (root)...")
    env._tap_element(locator_tap) # Tap within the subtree of current node (root)
    time.sleep(2) # Wait to observe effect

    # Try to input text (you'll need to navigate to a text field first in a real scenario)
    locator_input = {"class_name" : "android.widget.EditText"} # Example, may need adjustment
    text_to_input = "Hello from Current Node"
    print(f"Attempting to input text '{text_to_input}' in current node subtree...")
    env._input_text(locator_input, text_to_input) # Input text within current node subtree
    time.sleep(2)

    # Example of scrolling current element (if current node is scrollable)
    print("Attempting to scroll current element (if scrollable)...")
    env._scroll_current_element("down") # Scroll the current node element
    time.sleep(2)

    # Press back to reset UI for next test
    env._press_system_button("back")
    time.sleep(2)

