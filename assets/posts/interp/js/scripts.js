function showVisualizationGroup1(selectedValue) {
    // Hide all visualizations in group 1
    document.querySelectorAll('.visualization-group1').forEach(function(vis) {
      vis.style.display = 'none';
    });
  
    // Remove 'active' class from all buttons in group 1
    document.querySelectorAll('.group1').forEach(function(btn) {
      btn.classList.remove('active');
    });
  
    // Show the selected visualization in group 1
    document.getElementById(selectedValue).style.display = 'block';
  
    // Add 'active' class to the clicked button in group 1
    document.getElementById('btn_' + selectedValue).classList.add('active');
  }
  
  function showVisualizationGroup2(selectedValue) {
    // Hide all visualizations in group 2
    document.querySelectorAll('.visualization-group2').forEach(function(vis) {
      vis.style.display = 'none';
    });
  
    // Remove 'active' class from all buttons in group 2
    document.querySelectorAll('.group2').forEach(function(btn) {
      btn.classList.remove('active');
    });
  
    // Show the selected visualization in group 2
    document.getElementById(selectedValue).style.display = 'block';
  
    // Add 'active' class to the clicked button in group 2
    document.getElementById('btn_' + selectedValue).classList.add('active');
  }
  
  document.addEventListener('DOMContentLoaded', function() {
    // Automatically display the first visualization for both groups when the page loads
    showVisualizationGroup1('induction_3_0');
    showVisualizationGroup2('induction_3_3');
  });
  