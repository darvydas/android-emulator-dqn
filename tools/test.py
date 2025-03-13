import xml.etree.ElementTree as ET

xml_string = "<root><child><grandchild/></child></root>"
root = ET.fromstring(xml_string)
child = root[0]
grandchild = child[0]

print(child.getparent() == root) # Should be True
print(grandchild.getparent() == child) # Should be True
print(root.getparent() is None) # Should be True


# ~/android-studio/bin/studio.sh

# adb devices -l
# appium
# adb shell dumpsys window | grep -E 'mCurrentFocus

# /home/darv/Appium\ Inspector/run_appium_inspector.sh