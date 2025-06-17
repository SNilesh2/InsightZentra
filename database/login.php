<?php
session_start(); // Start the session

// Database connection
$servername = "localhost";
$username = "root";      // Default XAMPP username
$password = "";          // Default XAMPP password
$dbname = "article_summarizer";

$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Collect login form data
$email = $_POST['email'];
$password = $_POST['password'];

// Check if the email exists
$sql = "SELECT * FROM users WHERE email=?";
$stmt = $conn->prepare($sql);
$stmt->bind_param("s", $email);
$stmt->execute();
$result = $stmt->get_result();

// Verify user
if ($result->num_rows > 0) {
    $row = $result->fetch_assoc();
    
    // Verify password
    if (password_verify($password, $row['password'])) {
        // Password correct, set session
        $_SESSION['user_id'] = $row['id'];
        $_SESSION['fullname'] = $row['fullname'];
        $_SESSION['email'] = $row['email'];
        
        // Redirect to Flask app
        echo "<script>alert('Login successful!'); window.location.href='http://localhost:5000/';</script>";
    } else {
        echo "<script>alert('Incorrect password!'); window.location.href='login_signup.html';</script>";
    }
} else {
    echo "<script>alert('No account found with that email! Please signup.'); window.location.href='login_signup.html';</script>";
}

$conn->close();
?>
