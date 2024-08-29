from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

# Define your login credentials
username = "22111344@viit.ac.in"
password = "vidyaniketan@19"

# Initialize the WebDriver
driver = webdriver.Chrome()

try:
    # Open the login page
    driver.get("https://tpo.vierp.in")

    # Wait for the username field to be present
    wait = WebDriverWait(driver, 20)  # Increased wait time
    username_field = wait.until(EC.presence_of_element_located((By.ID, "input-15")))
    print("Username field located")

    # Find the password field
    password_field = driver.find_element(By.ID, "input-18")
    print("Password field located")

    # Input the credentials
    username_field.send_keys(username)
    password_field.send_keys(password)
    print("Credentials entered")

    # Wait for the login button to be clickable
    login_button = wait.until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'logi')]"))
    )
    print("Login button located")

    # Click the login button
    login_button.click()
    print("Login button clicked")

    # Wait for the homepage to load by checking the URL change
    wait.until(EC.url_contains("/home"))
    print("Home page loaded")

    # Navigate to the apply_company page
    driver.get("https://tpo.vierp.in/apply_company")
    print("Navigated to apply_company page")

    # Wait for the page to load and check if there is a company to apply
    try:
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "company")))
        print("Company found")

        # Find the apply button
        apply_button = driver.find_element(By.XPATH, "//button[text()='Apply']")
        # Click the apply button if it exists
        if apply_button:
            apply_button.click()
            print("Apply button clicked")
    except TimeoutException:
        print("No company to apply to.")

    # Optionally, you can wait and verify the application was successful
    time.sleep(5)

except NoSuchElementException as e:
    print(f"Element not found: {e}")
except TimeoutException as e:
    print(f"Loading timeout: {e}")
finally:
    # Close the WebDriver
    driver.quit()
