<?php
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

// Collect form data
$fullname = $_POST['fullname'];
$email = $_POST['email'];
$password = $_POST['password'];
$confirm_password = $_POST['confirm_password'];

// Check if passwords match
if ($password != $confirm_password) {
    echo "<script>alert('Passwords do not match!'); window.location.href='login_signup.html';</script>";
    exit();
}

// Check if email already exists
$checkEmail = "SELECT * FROM users WHERE email=?";
$stmt = $conn->prepare($checkEmail);
$stmt->bind_param("s", $email);
$stmt->execute();
$result = $stmt->get_result();

if ($result->num_rows > 0) {
    echo "<script>alert('Email already registered. Try logging in.'); window.location.href='login_signup.html';</script>";
    exit();
}

// Hash the password before storing
$hashed_password = password_hash($password, PASSWORD_DEFAULT);

// Insert user into database
$insert = "INSERT INTO users (fullname, email, password) VALUES (?, ?, ?)";
$stmt = $conn->prepare($insert);
$stmt->bind_param("sss", $fullname, $email, $hashed_password);

if ($stmt->execute()) {
    echo "<script>alert('Signup successful! Please login.'); window.location.href='login_signup.html';</script>";
} else {
    echo "Error: " . $stmt->error;
}

$conn->close();
?>
