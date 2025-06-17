function login() {
    document.getElementById("login").classList.add("active");
    document.getElementById("register").classList.remove("active");
}

function register() {
    document.getElementById("register").classList.add("active");
    document.getElementById("login").classList.remove("active");
}

function goFurther() {
    document.getElementById("btnSubmit").disabled = !document.getElementById("chkAgree").checked;
}
