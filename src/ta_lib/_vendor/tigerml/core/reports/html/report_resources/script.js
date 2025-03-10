$(document).ready( function () {
    datatables = $('.apply_datatable table').DataTable();
    $('.side_nav li.group > a').click(function() {
        $(this).toggleClass('closed');
    });
    UNSAVED_CHANGES = false;
    function bokeh_cleanup() {
        for (prop in Bokeh.index) {
            if(Bokeh.index.hasOwnProperty(prop)) {
                console.log("Removing bokeh plot: " + prop);
                Bokeh.index[prop].remove();
            }
            delete(Bokeh.documents);
        }
    }
    function bokeh_reinit() {
        window.Bokeh = Bokeh;
        $('[data-root-id]').each(function() {
            scripts = $(this).closest('.content_inner').first().find('script');
            window.eval(scripts[0].innerHTML);
            window.eval(scripts[1].innerHTML);
        });
//        bokeh_defaults();
    }
    function confirm_deletion(id, name) {
        $('#confirm_delete').find('.name').first().text(name);
        $('#confirm_delete').find('.confirm').first().attr('data-id', id);
        $('#confirm_delete').addClass('is_visible');
        $('#confirm_delete .confirm').focus();
    }
    function clean_parent(parent) {
        if (parent.find('.component').length == 0) {
            if (parent.hasClass('dashboard')) {
                delete_content_and_link(parent.attr('id'))
            } else {
                component = parent.closest('.component').first();
                if (component.attr('id')) {
                    delete_content_and_link(component.attr('id'))
                } else {
                    parent = get_parent(component);
                    component.remove();
                    if (parent) {
                        clean_parent(parent);
                    }
                }
            }
        }
    }
    function get_parent(component) {
        if (component.closest('.component_group').length > 0) {
            parent = component.closest('.component_group').first();
        } else {
            parent = component.closest('.dashboard').first();
        }
        if (parent == component || parent.length == 0) {
            return null;
        } else {
            return parent;
        }
    }
    function delete_content_and_link(id) {
        current_component = $('#'+id);
        parent = get_parent(current_component);
        current_component.remove()
        $('.side_nav').find("#nav_to_" + id).first().parent('li').first().remove();
        if (parent) {
            clean_parent(parent);
        }
    }
    function save_changes() {
        datatables.destroy();
        bokeh_cleanup();
        var blob = new Blob([$("html").html()], {type: "text/html;charset=utf-8"});
        file_path_bits = document.URL.split('#')[0].split('/')
        file_name = file_path_bits[file_path_bits.length - 1]
        new_file = file_name.substring(0, file_name.length-5) + '_modified.html'
        saveAs(blob, new_file);
        datatables = $('.table_content table').DataTable();
        bokeh_reinit()
        $('#report_saved').addClass('is_visible');
    }
    function check_for_changes() {
        if (UNSAVED_CHANGES) {
            $('#save_changes').addClass('is_visible');
        } else {
            $('#save_changes').removeClass('is_visible');
        }
    }
    $('.component[id]').filter(function () {
        return this.id != "";
    }).append("<span class='delete_component'></span>");
    $('.dashboard[id]').append("<span class='delete_dashboard'></span>");
    $('.delete_dashboard').click(function() {
        id = $(this).parent('.dashboard')[0].id
        name = $($(this).parent('.dashboard')[0]).find('.component_title').first().text()
        confirm_deletion(id, name)
    });
    $('.delete_component').click(function() {
        id = $(this).parent('.component')[0].id
        name = $($(this).parent('.component')[0]).find('.component_title').first().text()
        confirm_deletion(id, name)
    });
    $('#save_changes .confirm').click(function() {
        $('#save_changes').removeClass('is_visible');
        save_changes();
        UNSAVED_CHANGES = false;
        check_for_changes();
    });
    $('#confirm_delete .confirm').click(function() {
        id = $(this).attr('data-id')
        delete_content_and_link(id);
        UNSAVED_CHANGES = true;
        check_for_changes();
        $('#confirm_delete').removeClass('is_visible');
    });
    $('.glass').click(function() {
        $(this).parent('.overlay').first().removeClass('is_visible');
    });
    $('.cancel').click(function() {
        $(this).closest('.banner').first().removeClass('is_visible');
        $(this).closest('.overlay').first().removeClass('is_visible');
    });
    $('.overlay, .banner').removeClass('is_visible');
});

