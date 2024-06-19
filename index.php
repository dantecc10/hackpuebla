<?php
echo (isset($_GET['name']) && isset($_GET['last_names'])) ? ("Hola " . $_GET['name'] . " " . $_GET['last_names']) : 'No proporcionaste suficientes datos.';
